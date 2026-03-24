"""
Main script for LLM fine-tuning with LoRA.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main fine-tuning pipeline."""
    print("="*60)
    print("LLM FINE-TUNING WITH LoRA")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if device == "cpu":
        print("\nWARNING: Training on CPU is very slow. GPU recommended.")
        print("This demo will show the setup without full training.\n")
    
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT_DIR = str(Path(__file__).parent.parent / "adapters")
    USE_4BIT = device == "cuda"
    
    print("\n1. Creating sample dataset...")
    from src.dataset import create_sample_dataset, preprocess_dataset
    dataset = create_sample_dataset()
    print(f"   Dataset size: {len(dataset)} examples")
    
    print(f"\n2. Loading model: {MODEL_NAME}")
    try:
        from src.model import get_model_and_tokenizer, setup_lora, get_training_args, save_adapter
        
        model, tokenizer = get_model_and_tokenizer(
            MODEL_NAME,
            use_4bit=USE_4BIT,
            device_map="auto" if device != "cpu" else None
        )
        print("   Model loaded successfully!")
        
        print("\n3. Setting up LoRA adapters...")
        model = setup_lora(model, r=8, alpha=16, dropout=0.1)
        
        print("\n4. Preprocessing dataset...")
        processed_dataset = preprocess_dataset(dataset, tokenizer, max_length=512)
        print(f"   Processed {len(processed_dataset)} examples")
        
        if device != "cpu":
            print("\n5. Starting fine-tuning...")
            from trl import SFTTrainer
            
            training_args = get_training_args(OUTPUT_DIR, num_epochs=1, batch_size=1)
            
            trainer = SFTTrainer(
                model=model,
                train_dataset=processed_dataset,
                tokenizer=tokenizer,
                args=training_args,
                max_seq_length=512,
            )
            
            trainer.train()
            
            print("\n6. Saving adapter...")
            save_adapter(model, OUTPUT_DIR)
        else:
            print("\n5. Skipping training on CPU (too slow for demo)")
            print("   In production, use GPU/TPU for training.")
            
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            with open(Path(OUTPUT_DIR) / "training_skipped.txt", "w") as f:
                f.write("Training was skipped because no GPU was available.\n")
                f.write("Run on a machine with CUDA GPU for actual training.\n")
        
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Some dependencies may not be installed.")
        print("Install with: pip install -r requirements.txt")
        return None, None
    except Exception as e:
        print(f"\nError during setup: {e}")
        print("This is expected if running without GPU or on limited hardware.")
        return None, None
    
    print("\n" + "="*60)
    print("Fine-tuning pipeline complete!")
    print("="*60)
    
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, input_text: str = ""):
    """Generate a response from the fine-tuned model."""
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip()
    
    return response


if __name__ == "__main__":
    model, tokenizer = main()
    
    if model is not None:
        print("\n" + "="*60)
        print("TESTING GENERATION")
        print("="*60)
        
        test_prompts = [
            ("Explain what a neural network is.", ""),
            ("Write a simple Python hello world program.", ""),
        ]
        
        for instruction, input_text in test_prompts:
            print(f"\nInstruction: {instruction}")
            if input_text:
                print(f"Input: {input_text}")
            
            try:
                response = generate_response(model, tokenizer, instruction, input_text)
                print(f"Response: {response[:300]}...")
            except Exception as e:
                print(f"Generation error: {e}")
