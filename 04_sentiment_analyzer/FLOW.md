# End-to-End Flow: Sentiment Analyzer (LSTM vs BERT)

## Overview

This document explains the complete pipeline for sentiment classification on IMDB movie reviews, comparing two approaches: traditional LSTM and modern transformer-based BERT.

---

## Flow Diagram

```
IMDB Movie Reviews Dataset (50,000 reviews)
          │
          ▼
┌──────────────────────┐
│   Data Loading       │  ← HuggingFace datasets library
│   (Positive/Negative)│    or synthetic fallback
└──────────┬───────────┘
           │
     ┌─────┴─────────────────────┐
     │                           │
     ▼                           ▼
 LSTM Path                  BERT Path
     │                           │
     ▼                           ▼
Build Vocabulary           Load Pretrained
(word → int)               DistilBERT Tokenizer
     │                           │
     ▼                           ▼
LSTMDataset                IMDBDataset
(word indices)             (input_ids,
                            attention_mask)
     │                           │
     ▼                           ▼
┌──────────────┐         ┌──────────────────┐
│ LSTM Model   │         │ DistilBERT Model  │
│ Bidirectional│         │ + Custom Head     │
└──────┬───────┘         └────────┬─────────┘
       │                          │
       ▼                          ▼
 Train 5 epochs            Train 3 epochs
 LR = 0.001                LR = 2e-5 (low!)
       │                          │
       ▼                          ▼
   ~85% accuracy               ~92% accuracy
       │                          │
       └─────────┬────────────────┘
                 │
                 ▼
          Side-by-side comparison
          + Model saved to disk
```

---

## Step-by-Step Breakdown

### Step 1: Dataset
**File**: `src/dataset.py` → `load_imdb_data()`

- **IMDB Reviews**: 50,000 movie reviews (25k train, 25k test)
- Binary labels: 1 = Positive, 0 = Negative
- Average review length: ~230 words

```python
train_texts, train_labels, test_texts, test_labels = load_imdb_data(sample_size=2000)
```

If HuggingFace datasets unavailable, synthetic examples are generated for demo purposes.

---

### Step 2A: LSTM Path — Tokenization
**File**: `src/dataset.py` → `build_vocab()`, `LSTMDataset`

LSTM needs text converted to integer sequences:

```
"This movie was great!"
        │
        ▼  Lowercase + split
["this", "movie", "was", "great"]
        │
        ▼  Vocabulary lookup
    [42,    301,   27,   856]
        │
        ▼  Pad to max_length=256
[42, 301, 27, 856, 0, 0, 0, ..., 0]
```

**Vocabulary building**:
```
Count all words across training corpus
Keep top 30,000 words with frequency ≥ 2
Add special tokens: <PAD>=0, <UNK>=1
```

---

### Step 2B: BERT Path — Tokenization
**File**: `src/dataset.py` → `IMDBDataset`

BERT uses **subword tokenization** (WordPiece):

```
"This movie was great!"
        │
        ▼  DistilBERT tokenizer
["[CLS]", "this", "movie", "was", "great", "!", "[SEP]"]
        │
        ▼  Convert to token IDs
[101, 2023, 3185, 2001, 2307, 999, 102, 0, 0, ..., 0]
        │
        ▼  Attention mask (1=real, 0=padding)
[1,    1,    1,     1,    1,    1,   1,  0, 0, ..., 0]
```

**Why subword tokenization is better**:
- Handles unseen words ("unbelievable" → "un", "##believe", "##able")
- Same vocabulary works for all languages
- No out-of-vocabulary problem

---

### Step 3A: LSTM Model
**File**: `src/model.py` → `LSTMClassifier`

```
Input: [batch, seq_len] integer indices
   │
   ▼  Embedding(vocab_size, 128)
   │  Converts integers → dense vectors
   │  Output: [batch, seq_len, 128]
   │
   ▼  Bidirectional LSTM(128→256, 2 layers)
   │  Reads sequence forward AND backward
   │  Output: hidden_state [batch, 512]
   │            (concat of forward + backward final hidden)
   │
   ▼  Dropout(0.5)
   │
   ▼  FC(512→256) → ReLU → Dropout(0.5) → FC(256→2)
   │
   ▼  Output: [batch, 2] (logits for neg/pos)
```

**Why bidirectional?**
- "Not bad at all" — forward LSTM sees "not" before "bad"
- Backward LSTM sees "all" before "bad"
- Together they understand negation and context better

**LSTM internals**:
```
At each step t:
  forget_gate = σ(W_f · [h_{t-1}, x_t])   ← What to forget
  input_gate  = σ(W_i · [h_{t-1}, x_t])   ← What to add
  cell_state  = forget * C_{t-1} + input * tanh(...)
  output      = σ(W_o · [h_{t-1}, x_t]) * tanh(C_t)
```

---

### Step 3B: DistilBERT Model
**File**: `src/model.py` → `DistilBERTClassifier`

```
Input: input_ids [batch, 256], attention_mask [batch, 256]
   │
   ▼  DistilBERT-base-uncased (6 transformer layers)
   │  Self-attention: every token attends to every other token
   │  Output: [batch, 256, 768] (768-dim per token)
   │
   ▼  Take [CLS] token embedding: [batch, 768]
   │  CLS token aggregates the full sequence meaning
   │
   ▼  Dropout(0.3) → FC(768→384) → ReLU → Dropout(0.3) → FC(384→2)
   │
   ▼  Output: [batch, 2]
```

**Self-attention mechanism**:
```
For each token, compute:
  Q = token query,  K = all keys,  V = all values
  Attention(Q, K, V) = softmax(QK^T / √d_k) · V
  
"great" can directly attend to "not" in "not great"
Unlike LSTM, no sequential bottleneck!
```

---

### Step 4: Training

**LSTM Training**:
```python
optimizer = AdamW(lr=0.001)
criterion = CrossEntropyLoss()
# 5 epochs, batch_size=16
# Full pass: forward → loss → backward → step
```

**BERT Training**:
```python
optimizer = AdamW(lr=2e-5)  # 50× smaller LR to preserve pretrained weights
criterion = CrossEntropyLoss()
# 3 epochs, batch_size=16
# Lower LR prevents "catastrophic forgetting"
```

---

### Step 5: Evaluation

Both models are evaluated per epoch:

```
For each batch in test_loader:
  output = model(inputs)
  predictions = argmax(output, dim=1)
  
Compute:
  Accuracy  = correct / total
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 × (P × R) / (P + R)
```

| Model | Accuracy | Speed | Memory |
|-------|----------|-------|--------|
| LSTM | ~85% | Fast | Low |
| DistilBERT | ~92% | Slow | High |

---

### Step 6: Model Saving

```
models/
├── lstm_best.pth         ← {'model_state_dict', 'optimizer_state_dict', 'accuracy'}
└── distilbert_best.pth   ← best checkpoint from training
```

---

## Why BERT Beats LSTM

| Aspect | LSTM | BERT |
|--------|------|------|
| Context window | Sequential, limited | Full bidirectional attention |
| Pre-training | None | 110M params on huge corpus |
| Word representation | Static | Contextual (same word, different meaning) |
| "bank" in "river bank" | Same vector always | Different from "money bank" |
| Vanishing gradient | Problem in long sequences | Native support via attention |

---

## Data Flow Through Code

```
IMDB texts
  ├── LSTM path:
  │     build_vocab() → {word: int}
  │         └── LSTMDataset → (tensor_indices, label)
  │               └── DataLoader → batches
  │                     └── LSTMClassifier.forward()
  │                           └── loss → backward → step
  │
  └── BERT path:
        AutoTokenizer.from_pretrained('distilbert-base-uncased')
            └── IMDBDataset → {input_ids, attention_mask, label}
                  └── DataLoader → batches
                        └── DistilBERTClassifier.forward()
                              └── loss → backward → step
```

---

## Running the Full Flow

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train both models
python src/main.py

# Expected output:
# Training LSTM...
#   Epoch 1: Train Acc 0.72, Test Acc 0.78
#   ...
#   Epoch 5: Train Acc 0.89, Test Acc 0.85
#
# Training DistilBERT...
#   Epoch 1: Train Acc 0.85, Test Acc 0.89
#   Epoch 2: Train Acc 0.92, Test Acc 0.91
#   Epoch 3: Train Acc 0.95, Test Acc 0.92
```

---

## Key Learnings

| Concept | Where Used |
|---------|-----------|
| Word embeddings | LSTM: learned from scratch |
| Subword tokenization | BERT: handles OOV words |
| Bidirectional LSTM | Context in both directions |
| Self-attention | BERT's core mechanism |
| Transfer learning in NLP | Pretrained BERT fine-tuned |
| Learning rate for fine-tuning | 2e-5 to avoid catastrophic forgetting |
