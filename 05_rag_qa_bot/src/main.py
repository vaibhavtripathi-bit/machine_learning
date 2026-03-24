"""
Main script for RAG Q&A Bot.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_chain import RAGChain


def main():
    """Main RAG pipeline."""
    print("="*60)
    print("RAG Q&A BOT")
    print("="*60)
    
    print("\n1. Loading documents...")
    loader = DocumentLoader(chunk_size=500, chunk_overlap=100)
    
    docs_dir = Path(__file__).parent.parent / 'data' / 'documents'
    if docs_dir.exists() and list(docs_dir.glob('*')):
        documents = loader.load_directory(str(docs_dir))
        print(f"   Loaded {len(documents)} chunks from {docs_dir}")
    else:
        documents = loader.create_sample_documents()
        print(f"   Created {len(documents)} sample document chunks")
    
    print("\n2. Creating vector store...")
    persist_dir = str(Path(__file__).parent.parent / 'data' / 'vectorstore')
    vector_store = VectorStore(
        collection_name="rag_demo",
        persist_directory=persist_dir
    )
    vector_store.create_from_documents(documents)
    
    print("\n3. Testing retrieval...")
    test_query = "What is machine learning?"
    results = vector_store.similarity_search(test_query, k=2)
    print(f"   Query: '{test_query}'")
    print(f"   Found {len(results)} relevant chunks")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] {doc.page_content[:100]}...")
    
    print("\n4. Creating RAG chain...")
    rag_chain = RAGChain(vector_store)
    rag_chain.create_chain()
    print("   RAG chain ready!")
    
    print("\n5. Testing Q&A...")
    test_questions = [
        "What are the three types of machine learning?",
        "What is RAG and how does it work?",
        "What are transformers in deep learning?",
    ]
    
    for question in test_questions:
        print(f"\n   Q: {question}")
        result = rag_chain.query(question)
        print(f"   A: {result['answer'][:200]}...")
    
    print("\n" + "="*60)
    print("RAG Q&A Bot ready!")
    print("="*60)
    print("\nTo use with real OpenAI responses, set:")
    print("  export OPENAI_API_KEY='your-api-key'")
    
    return rag_chain, vector_store


def interactive_mode(rag_chain):
    """Run interactive Q&A mode."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("Type 'quit' to exit, 'clear' to reset memory")
    print("="*60)
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if question.lower() == 'quit':
                print("Goodbye!")
                break
            elif question.lower() == 'clear':
                rag_chain.clear_memory()
                print("Memory cleared.")
                continue
            elif not question:
                continue
            
            result = rag_chain.query(question)
            print(f"\nBot: {result['answer']}")
            
            if result['sources']:
                print("\nSources:")
                for i, source in enumerate(result['sources'][:2]):
                    print(f"  [{i+1}] {source['metadata'].get('source', 'Unknown')}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    rag_chain, vector_store = main()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode(rag_chain)
