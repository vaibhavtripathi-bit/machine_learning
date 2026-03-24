"""
RAG chain module.
Implements the retrieval-augmented generation pipeline.
"""

from typing import List, Dict, Optional
import os


class RAGChain:
    """RAG chain for question answering."""
    
    def __init__(
        self,
        vector_store,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0
    ):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store: VectorStore instance
            model_name: OpenAI model name
            temperature: Temperature for generation
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.chain = None
        self.memory = []
        
    def _get_llm(self):
        """Get the LLM instance."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature
            )
        else:
            print("No OpenAI API key found. Using mock LLM for demonstration.")
            return MockLLM()
    
    def create_chain(self):
        """Create the RAG chain."""
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser
        
        template = """You are a helpful AI assistant. Answer the question based on the following context.
If you cannot find the answer in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = self._get_llm()
        retriever = self.vector_store.get_retriever({"k": 4})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return self.chain
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and sources
        """
        if self.chain is None:
            self.create_chain()
        
        relevant_docs = self.vector_store.similarity_search(question, k=4)
        
        answer = self.chain.invoke(question)
        
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in relevant_docs
        ]
        
        self.memory.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    
    def query_with_history(self, question: str) -> Dict:
        """Query with conversation history."""
        history_context = ""
        if self.memory:
            recent = self.memory[-3:]
            history_context = "\n".join([
                f"Q: {m['question']}\nA: {m['answer']}"
                for m in recent
            ])
        
        enhanced_question = f"Previous conversation:\n{history_context}\n\nNew question: {question}" if history_context else question
        
        return self.query(enhanced_question)
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory = []


class MockLLM:
    """Mock LLM for demonstration without API key."""
    
    def invoke(self, prompt):
        """Return a mock response."""
        return "This is a mock response. To get real answers, please set the OPENAI_API_KEY environment variable."
    
    def __or__(self, other):
        """Support pipe operator for LangChain."""
        return ChainWrapper(self, other)


class ChainWrapper:
    """Wrapper to support LangChain pipe operations."""
    
    def __init__(self, llm, next_step):
        self.llm = llm
        self.next_step = next_step
    
    def invoke(self, input_data):
        result = self.llm.invoke(input_data)
        if hasattr(self.next_step, 'invoke'):
            return self.next_step.invoke(result)
        elif callable(self.next_step):
            return self.next_step(result)
        return result
