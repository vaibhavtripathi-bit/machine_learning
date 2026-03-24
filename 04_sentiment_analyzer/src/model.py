"""
Model module for sentiment analysis.
Implements LSTM and BERT-based classifiers.
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMClassifier(nn.Module):
    """LSTM-based sentiment classifier."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        num_classes: int = 2
    ):
        """
        Initialize the LSTM classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            num_classes: Number of output classes
        """
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        embedded = self.embedding(x)
        
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, _) = self.lstm(packed)
        else:
            lstm_out, (hidden, _) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        output = self.classifier(hidden)
        return output


class BERTClassifier(nn.Module):
    """BERT-based sentiment classifier."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        """
        Initialize the BERT classifier.
        
        Args:
            model_name: Name of pretrained BERT model
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT parameters
        """
        super(BERTClassifier, self).__init__()
        
        from transformers import AutoModel
        
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(pooled_output)
        return logits


class DistilBERTClassifier(nn.Module):
    """DistilBERT-based sentiment classifier (faster than BERT)."""
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super(DistilBERTClassifier, self).__init__()
        
        from transformers import AutoModel
        
        self.distilbert = AutoModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        hidden_size = self.distilbert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits


def get_model(
    model_type: str = 'distilbert',
    vocab_size: int = 30000,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Get a model for sentiment analysis.
    
    Args:
        model_type: Type of model ('lstm', 'bert', 'distilbert')
        vocab_size: Vocabulary size (for LSTM)
        num_classes: Number of output classes
        **kwargs: Additional model arguments
        
    Returns:
        PyTorch model
    """
    if model_type == 'lstm':
        return LSTMClassifier(vocab_size, num_classes=num_classes, **kwargs)
    elif model_type == 'bert':
        return BERTClassifier(num_classes=num_classes, **kwargs)
    elif model_type == 'distilbert':
        return DistilBERTClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
