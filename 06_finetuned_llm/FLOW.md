# End-to-End Flow: Fine-tuned LLM with LoRA

## Overview

This document explains the complete pipeline for fine-tuning a Large Language Model on custom instruction data using LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA).

---

## Flow Diagram

```
Custom Instruction Dataset
(instruction, input, output pairs)
          │
          ▼
┌──────────────────────┐
│  Format Prompts      │  ← Alpaca instruction template
│  (Alpaca format)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Tokenize            │  ← AutoTokenizer (subword)
│  (max_length=512)    │    Pad to fixed length
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Load Base LLM                       │
│  TinyLlama-1.1B (or Mistral-7B)      │
│                                      │
│  With QLoRA (4-bit quantization):    │
│  ┌──────────────────────────────┐    │
│  │  NF4 Quantization            │    │
│  │  Float32 → 4-bit NormalFloat │    │
│  │  Memory: 7B → ~4GB           │    │
│  └──────────────────────────────┘    │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Attach LoRA Adapters                │
│                                      │
│  Original: W (d×k)                   │
│  LoRA:     W + B×A  (r << d,k)       │
│            B: d×r,  A: r×k           │
│            rank r=8, only 0.5% params│
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────┐
│  SFT Training        │  ← Supervised Fine-Tuning
│  (TRL SFTTrainer)    │    AdamW + cosine LR
│                      │    gradient accumulation
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Save LoRA Adapter   │  ← Only adapter weights (~10MB)
│  (adapters/)         │    NOT the full model (~1.1GB)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Inference           │  ← Load base model + adapter
│  (generate)          │    model.generate() with sampling
└──────────────────────┘
```

---

## Step-by-Step Breakdown

### Step 1: Dataset Preparation
**File**: `src/dataset.py` → `create_sample_dataset()`, `format_instruction()`

Instructions are formatted in **Alpaca template**:

```
Without input:
──────────────────────────────────────────
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
Explain what machine learning is.

### Response:
Machine learning is a type of AI where systems learn from data...
──────────────────────────────────────────

With input:
──────────────────────────────────────────
### Instruction:
Summarize the following text.

### Input:
Neural networks are computing systems inspired by the brain...

### Response:
Neural networks are brain-inspired systems that learn from data.
──────────────────────────────────────────
```

**Why Alpaca format?** It's the industry standard for instruction tuning. Models learn to follow the `### Instruction` → `### Response` pattern.

---

### Step 2: QLoRA — Quantizing the Base Model
**File**: `src/model.py` → `get_model_and_tokenizer(use_4bit=True)`

Standard fine-tuning of a 7B model needs **~56GB GPU RAM**. QLoRA solves this:

```
Original weights (float32):  1 param = 4 bytes
                              7B params × 4 = 28GB
        │
        ▼  BitsAndBytes 4-bit NF4 quantization
        
NF4 weights (4-bit):         1 param = 0.5 bytes
                              7B params × 0.5 = 3.5GB  ← 8× reduction!
```

**NF4 (Normal Float 4)**: 16 quantization levels designed to match the normal distribution of neural network weights. More accurate than uniform quantization.

**Double quantization**: Quantizes the quantization constants themselves, saving additional ~0.37 bits/parameter.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16, # Compute in fp16
    bnb_4bit_use_double_quant=True       # Quantize quant constants too
)
```

---

### Step 3: LoRA — Efficient Fine-Tuning
**File**: `src/model.py` → `setup_lora()`

**The Problem**: We can't update 7B frozen quantized parameters.

**LoRA Solution**: Add small trainable matrices alongside frozen weights:

```
Forward pass (original):
  output = W · input           (W is frozen, d×k matrix)

Forward pass (LoRA):
  output = W · input + (B · A) · input · (alpha/r)
                    ↑
              trainable adapter
              B: d×r,  A: r×k
              r=8 << min(d,k)
```

**Parameter reduction**:
```
Original layer (d=4096, k=4096): 4096 × 4096 = 16.7M params
LoRA adapter (r=8):              (4096×8) + (8×4096) = 65K params

Reduction: 99.6% fewer parameters to train!
```

**Which layers get LoRA?**
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
#                  query    key       value     output
#                  ← All attention projection matrices →
```

Attention layers are targeted because they hold the most "knowledge" about language understanding.

---

### Step 4: Training with SFTTrainer
**File**: `src/main.py` → `main()`

**SFT = Supervised Fine-Tuning**: The model learns to predict the `### Response` given the `### Instruction`.

Training configuration:
```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,    # Effective batch = 4×4 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",       # LR anneals smoothly
    warmup_ratio=0.03,                # 3% of training for warmup
    optim="paged_adamw_8bit",         # Memory-efficient optimizer
    fp16=True,                        # Mixed precision training
)
```

**Gradient accumulation**: Accumulates gradients over 4 mini-batches before updating. Simulates a larger batch size without needing more memory.

---

### Step 5: Training Loop Internals

```
For each batch:
  1. Tokenized prompt → model forward pass
  2. Model predicts next token at each position
  3. Loss = CrossEntropy(predicted, actual) on response tokens only
     (instruction tokens are masked out — no loss computed there)
  4. Backward: gradients flow only through LoRA adapter weights
  5. Optimizer step: update B and A matrices
```

**Why mask instruction tokens?**
We only want the model to learn to generate good responses. Computing loss on the instruction would make the model try to "generate instructions" too, which isn't useful.

---

### Step 6: Saving & Loading Adapters
**File**: `src/model.py` → `save_adapter()`, `load_adapter()`

Only the tiny adapter is saved:
```
adapters/
├── adapter_config.json     ← LoRA config (r, alpha, target_modules)
├── adapter_model.bin       ← B and A matrices weights (~10-50MB)
└── tokenizer_config.json
```

This is the key advantage: you can share a 50MB adapter instead of a 14GB model.

**Loading at inference**:
```python
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama-1.1B")
model = PeftModel.from_pretrained(base_model, "./adapters")
# At inference, B×A is merged into W:
# W_merged = W_frozen + (B × A) × (alpha/r)
```

---

### Step 7: Inference
**File**: `src/main.py` → `generate_response()`

```
Instruction: "Explain what machine learning is."
        │
        ▼  Format as Alpaca prompt
        "Below is an instruction..."
        │
        ▼  Tokenize → input_ids
        [101, 342, ...]
        │
        ▼  model.generate()
        │  temperature=0.7 → diverse outputs
        │  top_p=0.9       → nucleus sampling
        │  max_new_tokens=256
        │
        ▼  Decode output_ids
        │
        ▼  Extract text after "### Response:"
        "Machine learning is a type of AI..."
```

---

## Memory Requirements

| Setup | GPU Memory | Training Time |
|-------|-----------|---------------|
| Full fine-tuning (fp32) | ~56GB | Baseline |
| LoRA (fp16) | ~14GB | ~2× faster |
| QLoRA (4-bit + LoRA) | ~6GB | ~3× faster |
| CPU (no GPU) | RAM only | Very slow |

---

## Running the Full Flow

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run (GPU will do full training, CPU shows setup only)
python src/main.py

# With GPU:
# → Loads TinyLlama 1.1B
# → Attaches LoRA (rank=8)
# → Trains on 8 sample examples
# → Saves adapter to adapters/
# → Tests generation
```

---

## Key Learnings

| Concept | Where Used |
|---------|-----------|
| QLoRA | 4-bit quantization for memory efficiency |
| LoRA rank | r=8 balances capacity vs parameter count |
| NF4 quantization | Better than uniform for neural weights |
| Instruction tuning | Alpaca format for task following |
| Gradient accumulation | Simulate larger batches on small GPU |
| Adapter saving | Share 50MB instead of 14GB |
| Catastrophic forgetting | Avoided by only updating adapter |
