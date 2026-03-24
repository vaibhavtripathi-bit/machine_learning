# End-to-End Flow: RAG Q&A Bot

## Overview

This document explains how the Retrieval-Augmented Generation (RAG) pipeline works — from loading documents to generating accurate, grounded answers.

---

## Flow Diagram

```
Documents (PDF, TXT, MD)
          │
          ▼
┌──────────────────────┐
│  Document Loading    │  ← DocumentLoader reads files
│  & Chunking          │    Split into ~500 char chunks
│                      │    with 100 char overlap
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Embedding           │  ← sentence-transformers
│  (Text → Vectors)    │    all-MiniLM-L6-v2
│                      │    Each chunk → 384-dim vector
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Vector Store        │  ← ChromaDB stores vectors
│  (ChromaDB)          │    Persisted to disk
│                      │
└──────────────────────┘


         QUERY TIME
          │
User Question: "What is RAG?"
          │
          ▼
┌──────────────────────┐
│  Question Embedding  │  ← Same embedding model
│                      │    Question → 384-dim vector
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Similarity Search   │  ← Cosine similarity in ChromaDB
│  (Top-K retrieval)   │    Returns top 4 relevant chunks
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Prompt Construction │  ← Template: Context + Question
│                      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LLM Generation      │  ← OpenAI GPT-3.5 / Mock LLM
│                      │    Generates grounded answer
└──────────┬───────────┘
           │
           ▼
   Answer + Sources
```

---

## Step-by-Step Breakdown

### Step 1: Document Loading & Chunking
**File**: `src/document_loader.py` → `DocumentLoader`

Raw documents are too long to fit in an LLM's context window. They must be split into smaller chunks.

```
"Machine learning is a subset of AI... [3000 words document]"
        │
        ▼  RecursiveCharacterTextSplitter
        │  chunk_size=500, chunk_overlap=100
        │
[Chunk 1: "Machine learning is a subset of AI that enables..."]
[Chunk 2: "...that enables systems to learn from examples..."]  ← 100 char overlap
[Chunk 3: "...from examples. There are three main types:..."]
```

**Why overlap?** Context at chunk boundaries might be split mid-sentence. Overlap ensures continuity.

**Splitting hierarchy** (tries in order, falls back):
1. `\n\n` — paragraph breaks
2. `\n` — line breaks
3. ` ` — word spaces
4. `""` — character-level (last resort)

---

### Step 2: Embedding
**File**: `src/vector_store.py` → `VectorStore._get_embeddings()`

Each chunk is converted to a dense vector representing its **semantic meaning**:

```
"What is machine learning?"  →  [0.12, -0.34, 0.89, ...]  (384 numbers)
"What is deep learning?"     →  [0.11, -0.31, 0.87, ...]  (very similar!)
"What is the weather today?" →  [0.78,  0.23, -0.45, ...]  (very different)
```

Model used: `all-MiniLM-L6-v2` (sentence-transformers)
- Fast and small (80MB)
- Produces 384-dimensional vectors
- Trained on semantic similarity tasks

**Similarity = Cosine Distance**:
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```
Values: -1 (opposite) to +1 (identical meaning)

---

### Step 3: Vector Store (ChromaDB)
**File**: `src/vector_store.py` → `VectorStore.create_from_documents()`

ChromaDB stores:
- The chunk text
- The embedding vector
- Metadata (source file, chunk index)

```
ChromaDB Collection: "rag_documents"
┌─────────────────────────────────────────────────────┐
│  ID  │ Embedding (384-dim) │ Text        │ Metadata │
│──────┼─────────────────────┼─────────────┼──────────│
│  0   │ [0.12, -0.34, ...]  │ "ML is..."  │ {src:.}  │
│  1   │ [0.11, -0.31, ...]  │ "DL is..."  │ {src:.}  │
│  2   │ [0.45,  0.22, ...]  │ "RAG is..." │ {src:.}  │
│  ...                                                 │
└─────────────────────────────────────────────────────┘
```

Persisted to disk: can be loaded without re-embedding on restart.

---

### Step 4: Query — Retrieval
**File**: `src/vector_store.py` → `VectorStore.similarity_search()`

At query time:

```
User: "What are the types of machine learning?"
         │
         ▼  Embed the question
         [0.14, -0.35, 0.90, ...]   ← question vector
         │
         ▼  Approximate Nearest Neighbor search in ChromaDB
         │  Computes cosine similarity with all stored chunks
         │
         ▼  Return Top-K=4 most similar chunks
         
Chunk 1: "There are three main types: supervised, unsupervised..."
Chunk 2: "Supervised learning uses labeled data..."
Chunk 3: "Unsupervised learning finds patterns..."
Chunk 4: "Reinforcement learning uses rewards..."
```

---

### Step 5: Prompt Construction
**File**: `src/rag_chain.py` → `RAGChain.query()`

Retrieved chunks + user question are combined into a structured prompt:

```
System: You are a helpful AI assistant. Answer based on context only.

Context:
[Chunk 1 text]

[Chunk 2 text]

[Chunk 3 text]

[Chunk 4 text]

Question: What are the types of machine learning?

Answer:
```

**Why structured prompts?**
- Prevents hallucination by grounding in retrieved context
- The `"If not in context, say 'I don't know'"` instruction prevents making things up
- Temperature=0.0 for factual, deterministic answers

---

### Step 6: LLM Generation
**File**: `src/rag_chain.py` → `RAGChain._get_llm()`

The LLM sees the prompt and generates an answer:

```
GPT-3.5 / DistilBERT / Mock LLM
         │
         ▼
"There are three main types of machine learning:
 1. Supervised learning, which uses labeled training data...
 2. Unsupervised learning, which finds hidden patterns...
 3. Reinforcement learning, which learns through rewards..."
```

**With OpenAI**: Set `OPENAI_API_KEY` env var  
**Without**: Mock LLM runs locally (for demo/testing)

---

### Step 7: Conversation Memory
**File**: `src/rag_chain.py` → `RAGChain.query_with_history()`

Memory stores the last 3 Q&A pairs and prepends them to each new query:

```
Previous conversation:
Q: What is machine learning?
A: Machine learning is a subset of AI...

Q: What are its types?
A: Supervised, unsupervised, reinforcement...

New question: Can you give an example of supervised learning?
```

This gives the bot **conversation context** without reloading the whole history.

---

## LangChain Pipeline

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
```

Data flow through the chain:
```
"What is RAG?"
      │
      ▼  retriever.invoke("What is RAG?")
      │  → [Document, Document, Document, Document]
      │
      ▼  format_docs()
      │  → "Doc1 text\n\nDoc2 text\n\n..."
      │
      ▼  prompt_template.invoke({context: ..., question: ...})
      │  → ChatPromptValue
      │
      ▼  llm.invoke(prompt)
      │  → AIMessage("RAG stands for Retrieval-Augmented...")
      │
      ▼  StrOutputParser()
      │
      ▼  "RAG stands for Retrieval-Augmented..."
```

---

## Running the Full Flow

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2 (optional): Set OpenAI key for real LLM
export OPENAI_API_KEY='your-key'

# Step 3: Run
python src/main.py

# Step 4: Interactive mode
python src/main.py --interactive

# Step 5: Add your own documents
cp your_docs/*.pdf data/documents/
python src/main.py
```

---

## Key Learnings

| Concept | Where Used |
|---------|-----------|
| Text chunking | RecursiveCharacterTextSplitter |
| Semantic embeddings | all-MiniLM-L6-v2 → 384-dim vectors |
| Vector similarity search | ChromaDB, cosine similarity |
| Prompt engineering | Grounding LLM to retrieved context |
| Hallucination mitigation | "Only answer from context" instruction |
| Conversation memory | Last-3-turns context window |
| LangChain LCEL | Composable pipeline with `|` operator |
