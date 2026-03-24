"""
Document loading and processing module.
Handles loading PDFs, text files, and chunking.
"""

from typing import List, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentLoader:
    """Load and process documents for RAG."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """Load a text file and split into chunks."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = self.text_splitter.split_text(text)
        
        return [
            Document(
                page_content=chunk,
                metadata={"source": file_path, "chunk_index": i}
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF file and split into chunks."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            documents = self.text_splitter.split_documents(pages)
            return documents
        except ImportError:
            print("pypdf not installed. Install with: pip install pypdf")
            return []
    
    def load_directory(self, directory_path: str, glob_pattern: str = "**/*") -> List[Document]:
        """Load all documents from a directory."""
        path = Path(directory_path)
        documents = []
        
        for file_path in path.glob(glob_pattern):
            if file_path.is_file():
                if file_path.suffix == '.txt':
                    documents.extend(self.load_text_file(str(file_path)))
                elif file_path.suffix == '.pdf':
                    documents.extend(self.load_pdf(str(file_path)))
                elif file_path.suffix == '.md':
                    documents.extend(self.load_text_file(str(file_path)))
        
        return documents
    
    def create_sample_documents(self) -> List[Document]:
        """Create sample documents for demonstration."""
        sample_texts = [
            """
            Machine Learning Basics
            
            Machine learning is a subset of artificial intelligence that enables systems to learn
            and improve from experience without being explicitly programmed. There are three main
            types of machine learning:
            
            1. Supervised Learning: The algorithm learns from labeled training data. Examples
               include classification (spam detection) and regression (price prediction).
            
            2. Unsupervised Learning: The algorithm finds patterns in unlabeled data. Examples
               include clustering (customer segmentation) and dimensionality reduction (PCA).
            
            3. Reinforcement Learning: The algorithm learns by interacting with an environment
               and receiving rewards or penalties. Used in game playing and robotics.
            """,
            """
            Deep Learning and Neural Networks
            
            Deep learning is a subset of machine learning based on artificial neural networks.
            Key concepts include:
            
            - Neurons: Basic units that receive inputs, apply weights, and produce outputs
            - Layers: Networks are organized into input, hidden, and output layers
            - Activation Functions: Non-linear functions like ReLU, sigmoid, and tanh
            - Backpropagation: Algorithm for training networks by computing gradients
            - Loss Functions: Measure how well the network performs (MSE, cross-entropy)
            
            Popular architectures include CNNs for images and RNNs/Transformers for sequences.
            """,
            """
            Large Language Models (LLMs)
            
            Large Language Models are neural networks trained on vast amounts of text data.
            Key characteristics:
            
            - Transformer Architecture: Uses self-attention mechanisms to process sequences
            - Pre-training: Models learn language patterns from large corpora
            - Fine-tuning: Models are adapted to specific tasks with smaller datasets
            - Prompt Engineering: Crafting inputs to get desired outputs
            - RAG (Retrieval-Augmented Generation): Combining LLMs with external knowledge
            
            Examples include GPT-4, Claude, LLaMA, and Mistral.
            """,
            """
            RAG (Retrieval-Augmented Generation)
            
            RAG is a technique that enhances LLM responses by retrieving relevant information
            from external sources. The process involves:
            
            1. Document Processing: Split documents into chunks
            2. Embedding: Convert chunks into vector representations
            3. Vector Storage: Store embeddings in a vector database
            4. Retrieval: Find relevant chunks for a given query
            5. Generation: Use retrieved context to generate accurate responses
            
            Benefits include reduced hallucination, up-to-date information, and domain expertise.
            Popular tools: LangChain, LlamaIndex, ChromaDB, Pinecone.
            """
        ]
        
        documents = []
        for i, text in enumerate(sample_texts):
            chunks = self.text_splitter.split_text(text.strip())
            for j, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source": f"sample_doc_{i}", "chunk_index": j}
                ))
        
        return documents
