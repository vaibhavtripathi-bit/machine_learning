"""
Dataset module for sentiment analysis.
Handles IMDB dataset loading and preprocessing.
"""

from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class IMDBDataset(Dataset):
    """IMDB dataset for sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        """
        Initialize the dataset.
        
        Args:
            texts: List of review texts
            labels: List of labels (0=negative, 1=positive)
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class LSTMDataset(Dataset):
    """Dataset for LSTM model with simple tokenization."""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int = 256):
        """
        Initialize the dataset.
        
        Args:
            texts: List of review texts
            labels: List of labels
            vocab: Vocabulary dictionary
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx].lower()
        label = self.labels[idx]
        
        tokens = text.split()[:self.max_length]
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def build_vocab(texts: List[str], min_freq: int = 2, max_vocab: int = 30000) -> Dict[str, int]:
    """
    Build vocabulary from texts.
    
    Args:
        texts: List of texts
        min_freq: Minimum frequency for a word to be included
        max_vocab: Maximum vocabulary size
        
    Returns:
        Vocabulary dictionary
    """
    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.lower().split())
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(max_vocab - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def collate_fn_lstm(batch):
    """Collate function for LSTM with padding."""
    texts, labels = zip(*batch)
    
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    lengths = torch.tensor([len(t) for t in texts])
    
    return texts_padded, labels, lengths


def load_imdb_data(sample_size: int = None) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Load IMDB dataset.
    
    Args:
        sample_size: Number of samples to load (None for all)
        
    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    try:
        from datasets import load_dataset
        
        print("Loading IMDB dataset from HuggingFace...")
        dataset = load_dataset('imdb')
        
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']
        
        if sample_size:
            train_texts = train_texts[:sample_size]
            train_labels = train_labels[:sample_size]
            test_texts = test_texts[:sample_size // 4]
            test_labels = test_labels[:sample_size // 4]
        
        print(f"Loaded {len(train_texts)} training samples, {len(test_texts)} test samples")
        return train_texts, train_labels, test_texts, test_labels
        
    except Exception as e:
        print(f"Error loading IMDB dataset: {e}")
        print("Creating synthetic dataset for demonstration...")
        
        positive_samples = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "A masterpiece of cinema. The acting was superb and the story was gripping.",
            "One of the best films I've ever seen. Highly recommended!",
            "Brilliant performances and a beautiful storyline. A must-watch.",
            "An incredible film that moved me to tears. Simply amazing.",
        ]
        
        negative_samples = [
            "This movie was terrible. I couldn't wait for it to end.",
            "A complete waste of time. The acting was awful and the plot made no sense.",
            "One of the worst films I've ever seen. Avoid at all costs.",
            "Boring, predictable, and poorly executed. Very disappointing.",
            "I want my two hours back. This movie was absolutely dreadful.",
        ]
        
        n_samples = sample_size or 1000
        train_texts = (positive_samples * (n_samples // 10)) + (negative_samples * (n_samples // 10))
        train_labels = [1] * (n_samples // 2) + [0] * (n_samples // 2)
        
        test_texts = (positive_samples * (n_samples // 40)) + (negative_samples * (n_samples // 40))
        test_labels = [1] * (n_samples // 8) + [0] * (n_samples // 8)
        
        import random
        combined_train = list(zip(train_texts, train_labels))
        random.shuffle(combined_train)
        train_texts, train_labels = zip(*combined_train)
        
        return list(train_texts), list(train_labels), list(test_texts), list(test_labels)


def get_dataloaders(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    tokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    model_type: str = 'bert'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and testing.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        test_texts: Test texts
        test_labels: Test labels
        tokenizer: Tokenizer (or vocab dict for LSTM)
        batch_size: Batch size
        max_length: Maximum sequence length
        model_type: 'bert' or 'lstm'
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if model_type == 'lstm':
        train_dataset = LSTMDataset(train_texts, train_labels, tokenizer, max_length)
        test_dataset = LSTMDataset(test_texts, test_labels, tokenizer, max_length)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_lstm
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_lstm
        )
    else:
        train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
        test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
