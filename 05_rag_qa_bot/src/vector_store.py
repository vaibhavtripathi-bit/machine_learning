"""
Vector store module for RAG.
Handles embedding and retrieval with ChromaDB.
"""

from typing import List, Optional
from pathlib import Path

from langchain.schema import Document


class VectorStore:
    """Vector store using ChromaDB for document retrieval."""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.vectorstore = None
        self._embeddings = None
        
    def _get_embeddings(self):
        """Get or create embeddings model."""
        if self._embeddings is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                print(f"Error loading HuggingFace embeddings: {e}")
                print("Falling back to simple embeddings...")
                from langchain_community.embeddings import FakeEmbeddings
                self._embeddings = FakeEmbeddings(size=384)
        return self._embeddings
    
    def create_from_documents(self, documents: List[Document]) -> None:
        """
        Create vector store from documents.
        
        Args:
            documents: List of Document objects
        """
        from langchain_community.vectorstores import Chroma
        
        embeddings = self._get_embeddings()
        
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        print(f"Created vector store with {len(documents)} documents")
    
    def load(self) -> None:
        """Load existing vector store from disk."""
        from langchain_community.vectorstores import Chroma
        
        if not self.persist_directory:
            raise ValueError("No persist directory specified")
        
        embeddings = self._get_embeddings()
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Loaded vector store from {self.persist_directory}")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_from_documents() or load() first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, search_kwargs: dict = None):
        """
        Get a retriever for use with LangChain chains.
        
        Args:
            search_kwargs: Arguments for similarity search
            
        Returns:
            LangChain retriever
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add more documents to existing store."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        self.vectorstore.add_documents(documents)
        print(f"Added {len(documents)} documents to vector store")
