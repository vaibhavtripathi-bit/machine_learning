"""
Dataset preparation for LLM fine-tuning.
"""

from typing import List, Dict
from datasets import Dataset


def create_instruction_dataset(examples: List[Dict[str, str]]) -> Dataset:
    """
    Create a dataset in instruction format.
    
    Args:
        examples: List of {"instruction": ..., "input": ..., "output": ...}
        
    Returns:
        HuggingFace Dataset
    """
    return Dataset.from_list(examples)


def format_instruction(example: Dict[str, str], tokenizer) -> str:
    """
    Format example in Alpaca instruction format.
    
    Args:
        example: Dictionary with instruction, input, output
        tokenizer: Tokenizer for special tokens
        
    Returns:
        Formatted prompt string
    """
    if example.get("input"):
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    
    return prompt


def create_sample_dataset() -> Dataset:
    """Create a sample dataset for demonstration."""
    examples = [
        {
            "instruction": "Explain what machine learning is in simple terms.",
            "input": "",
            "output": "Machine learning is a type of artificial intelligence where computers learn from examples rather than being explicitly programmed. Just like how you learn to recognize cats by seeing many pictures of cats, a machine learning model learns patterns from data to make predictions or decisions."
        },
        {
            "instruction": "Write a Python function that calculates the factorial of a number.",
            "input": "",
            "output": """def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)"""
        },
        {
            "instruction": "Summarize the following text.",
            "input": "Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes that process information using connectionist approaches to computation. Modern neural networks are capable of learning complex patterns from data.",
            "output": "Neural networks are brain-inspired computing systems made of connected nodes that learn patterns from data."
        },
        {
            "instruction": "Translate the following English text to French.",
            "input": "Hello, how are you today?",
            "output": "Bonjour, comment allez-vous aujourd'hui?"
        },
        {
            "instruction": "What are the benefits of using version control?",
            "input": "",
            "output": "Version control provides several key benefits: 1) Track changes history - see who changed what and when, 2) Collaboration - multiple people can work on the same project, 3) Backup - your code is safely stored, 4) Branching - experiment without affecting the main code, 5) Rollback - easily revert to previous versions if something breaks."
        },
        {
            "instruction": "Debug the following Python code.",
            "input": "def add(a, b):\n    return a + c",
            "output": "The bug is that the function uses 'c' instead of 'b'. Here's the corrected code:\n\ndef add(a, b):\n    return a + b"
        },
        {
            "instruction": "Explain the difference between supervised and unsupervised learning.",
            "input": "",
            "output": "Supervised learning uses labeled data where each example has an input and the correct output. The model learns to map inputs to outputs. Examples: classification, regression.\n\nUnsupervised learning uses unlabeled data where the model finds hidden patterns on its own. Examples: clustering, dimensionality reduction."
        },
        {
            "instruction": "Write a SQL query to find all users who signed up in the last 30 days.",
            "input": "Table: users (id, name, email, signup_date)",
            "output": "SELECT id, name, email, signup_date\nFROM users\nWHERE signup_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)\nORDER BY signup_date DESC;"
        },
    ]
    
    return create_instruction_dataset(examples)


def preprocess_dataset(dataset: Dataset, tokenizer, max_length: int = 512):
    """
    Preprocess dataset for training.
    
    Args:
        dataset: HuggingFace Dataset
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Preprocessed dataset
    """
    def tokenize(example):
        prompt = format_instruction(example, tokenizer)
        
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    return dataset.map(tokenize, remove_columns=dataset.column_names)
