"""
Model setup for LLM fine-tuning with LoRA/QLoRA.
"""

from typing import Optional
import torch


def get_model_and_tokenizer(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_4bit: bool = True,
    device_map: str = "auto"
):
    """
    Load model and tokenizer with optional quantization.
    
    Args:
        model_name: HuggingFace model name
        use_4bit: Whether to use 4-bit quantization
        device_map: Device mapping strategy
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    model.config.use_cache = False
    
    return model, tokenizer


def setup_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.1):
    """
    Setup LoRA adapters for the model.
    
    Args:
        model: Base model
        r: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout rate
        
    Returns:
        Model with LoRA adapters
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def get_training_args(output_dir: str, num_epochs: int = 3, batch_size: int = 4):
    """
    Get training arguments for SFT.
    
    Args:
        output_dir: Directory for outputs
        num_epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        TrainingArguments
    """
    from transformers import TrainingArguments
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )


def save_adapter(model, save_path: str):
    """Save LoRA adapter weights."""
    model.save_pretrained(save_path)
    print(f"Adapter saved to {save_path}")


def load_adapter(base_model_name: str, adapter_path: str):
    """Load a fine-tuned model with adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer
