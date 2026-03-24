# RAG Q&A Bot

A Retrieval-Augmented Generation (RAG) system that answers questions based on your documents using vector search and LLMs.

## Features

- **Document Processing**: Load PDFs, text files, and markdown
- **Smart Chunking**: Recursive text splitting with overlap
- **Vector Search**: ChromaDB for fast similarity search
- **LLM Integration**: OpenAI GPT models for answer generation
- **Conversation Memory**: Maintains context across questions
- **FastAPI Server**: REST API for production use

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (optional - works with mock LLM too)
export OPENAI_API_KEY='your-api-key'

# Run the bot
python src/main.py

# Interactive mode
python src/main.py --interactive
```

## Project Structure

```
05_rag_qa_bot/
├── data/
│   ├── documents/          # Your documents (PDFs, txt, md)
│   └── vectorstore/        # ChromaDB persistence
├── src/
│   ├── __init__.py
│   ├── document_loader.py  # Document loading and chunking
│   ├── vector_store.py     # ChromaDB vector store
│   ├── rag_chain.py        # RAG pipeline
│   └── main.py             # Main script
├── api/
│   └── app.py              # FastAPI server
├── notebooks/              # Jupyter notebooks
├── requirements.txt
└── README.md
```

## How RAG Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Documents  │────►│   Chunking  │────►│  Embedding  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◄────│     LLM     │◄────│  Retrieval  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                           ▲                    │
                           │                    ▼
                    ┌──────┴──────┐     ┌─────────────┐
                    │   Question  │     │ Vector Store│
                    └─────────────┘     └─────────────┘
```

## Key Concepts

### 1. Document Chunking
- Split documents into smaller pieces (500-1000 tokens)
- Use overlap (100-200 tokens) to preserve context
- Recursive splitting respects document structure

### 2. Embeddings
- Convert text to dense vectors (384-1536 dimensions)
- Semantic similarity through vector distance
- Uses sentence-transformers (all-MiniLM-L6-v2)

### 3. Vector Store
- ChromaDB for local vector storage
- Fast approximate nearest neighbor search
- Persistence for production use

### 4. Retrieval
- Find top-k similar chunks for a query
- Re-ranking for improved relevance
- Metadata filtering for specific sources

### 5. Generation
- Combine retrieved context with question
- LLM generates grounded answer
- Reduces hallucination through context

## Configuration

### Chunking Parameters
```python
loader = DocumentLoader(
    chunk_size=1000,    # Characters per chunk
    chunk_overlap=200   # Overlap between chunks
)
```

### Vector Store
```python
vector_store = VectorStore(
    collection_name="my_docs",
    persist_directory="./data/vectorstore",
    embedding_model="all-MiniLM-L6-v2"
)
```

### RAG Chain
```python
rag_chain = RAGChain(
    vector_store=vector_store,
    model_name="gpt-3.5-turbo",
    temperature=0.0
)
```

## Usage Examples

### Basic Usage
```python
from src.main import main
rag_chain, vector_store = main()

result = rag_chain.query("What is machine learning?")
print(result['answer'])
print(result['sources'])
```

### Add Custom Documents
```python
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore

loader = DocumentLoader()
docs = loader.load_directory("./my_documents")

vector_store = VectorStore(persist_directory="./my_vectorstore")
vector_store.create_from_documents(docs)
```

## Extending the Project

- Add web scraping for online sources
- Implement hybrid search (keyword + semantic)
- Add source citation in answers
- Deploy with FastAPI + Docker
- Add streaming responses

## License

MIT License
