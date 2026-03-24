# Fine-tuned LLM with LoRA

Fine-tune a Large Language Model on custom data using Parameter-Efficient Fine-Tuning (PEFT) with LoRA/QLoRA.

## Features

- **LoRA (Low-Rank Adaptation)**: Train only ~1% of parameters
- **QLoRA (Quantized LoRA)**: 4-bit quantization for memory efficiency
- **Instruction Tuning**: Alpaca-style instruction format
- **HuggingFace Integration**: Easy model sharing

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run fine-tuning (GPU required for full training)
python src/main.py
```

## Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **CUDA**: 11.8+ for bitsandbytes quantization
- For CPU: Demo mode only (no training)

## Project Structure

```
06_finetuned_llm/
├── data/                   # Training data
├── adapters/               # Saved LoRA adapters
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Dataset preparation
│   ├── model.py            # Model setup with LoRA
│   └── main.py             # Training pipeline
├── notebooks/              # Jupyter notebooks
├── requirements.txt
└── README.md
```

## How LoRA Works

```
Original Linear Layer:    W (d × k)
                         ↓
LoRA Decomposition:      W + BA
                         where B (d × r), A (r × k)
                         r << min(d, k)

Full fine-tuning:        Train all 7B parameters
LoRA fine-tuning:        Train only ~10M parameters (r=8)
```

### Key Benefits
1. **Memory Efficient**: 4-bit base model + small adapters
2. **Fast Training**: 10-100x fewer parameters to update
3. **Easy Sharing**: Adapters are only a few MB
4. **Composable**: Merge multiple adapters

## Configuration

### LoRA Parameters
```python
lora_config = LoraConfig(
    r=8,              # Rank (lower = smaller, higher = more capacity)
    lora_alpha=16,    # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### QLoRA Quantization
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # NormalFloat4
    bnb_4bit_compute_dtype=float16,
    bnb_4bit_use_double_quant=True  # Nested quantization
)
```

## Instruction Format

Using Alpaca-style prompts:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}
```

With input:
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

## Training

```python
from src.model import get_model_and_tokenizer, setup_lora
from src.dataset import create_sample_dataset, preprocess_dataset

# Load model
model, tokenizer = get_model_and_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Add LoRA
model = setup_lora(model, r=8, alpha=16)

# Train
trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
trainer.train()

# Save adapter
model.save_pretrained("./adapters")
```

## Inference

```python
from src.main import generate_response
from src.model import load_adapter

# Load fine-tuned model
model, tokenizer = load_adapter("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "./adapters")

# Generate
response = generate_response(model, tokenizer, "Explain machine learning")
```

## Key Concepts

1. **PEFT**: Parameter-Efficient Fine-Tuning reduces training cost
2. **LoRA Rank**: Controls adapter capacity (r=8 is common)
3. **Alpha/Rank Ratio**: Scaling factor (alpha=2*r is typical)
4. **Target Modules**: Which layers to adapt (attention is key)
5. **Instruction Tuning**: Format for task-following behavior

## Extending the Project

- Use larger base model (Mistral-7B, LLaMA-3-8B)
- Train on domain-specific data
- Push adapter to HuggingFace Hub
- Implement DPO/RLHF for alignment
- Create merged model for inference

## License

MIT License
