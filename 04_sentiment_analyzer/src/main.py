"""
Main script for sentiment analysis comparison.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_imdb_data, build_vocab, get_dataloaders
from src.model import get_model
from src.train import train_model


def main():
    """Main training pipeline comparing LSTM vs BERT."""
    print("="*60)
    print("SENTIMENT ANALYZER - LSTM vs BERT")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    NUM_EPOCHS_LSTM = 5
    NUM_EPOCHS_BERT = 3
    LEARNING_RATE_LSTM = 0.001
    LEARNING_RATE_BERT = 2e-5
    SAMPLE_SIZE = 2000
    
    print("\n1. Loading IMDB dataset...")
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(sample_size=SAMPLE_SIZE)
    
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)
    
    print("\n2. Building vocabulary...")
    vocab = build_vocab(train_texts, min_freq=2)
    print(f"   Vocabulary size: {len(vocab)}")
    
    print("\n3. Creating LSTM data loaders...")
    train_loader_lstm, test_loader_lstm = get_dataloaders(
        train_texts, train_labels, test_texts, test_labels,
        tokenizer=vocab, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, model_type='lstm'
    )
    
    print("\n4. Creating LSTM model...")
    lstm_model = get_model('lstm', vocab_size=len(vocab))
    total_params = sum(p.numel() for p in lstm_model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    print("\n5. Training LSTM...")
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    lstm_history = train_model(
        lstm_model, train_loader_lstm, test_loader_lstm,
        num_epochs=NUM_EPOCHS_LSTM, learning_rate=LEARNING_RATE_LSTM,
        device=device, model_type='lstm',
        save_path=str(models_dir / 'lstm_best.pth')
    )
    
    print("\n" + "="*60)
    print("TRAINING DISTILBERT MODEL")
    print("="*60)
    
    print("\n6. Loading DistilBERT tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    print("\n7. Creating BERT data loaders...")
    train_loader_bert, test_loader_bert = get_dataloaders(
        train_texts, train_labels, test_texts, test_labels,
        tokenizer=tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, model_type='bert'
    )
    
    print("\n8. Creating DistilBERT model...")
    bert_model = get_model('distilbert')
    total_params = sum(p.numel() for p in bert_model.parameters())
    trainable_params = sum(p.numel() for p in bert_model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n9. Training DistilBERT...")
    bert_history = train_model(
        bert_model, train_loader_bert, test_loader_bert,
        num_epochs=NUM_EPOCHS_BERT, learning_rate=LEARNING_RATE_BERT,
        device=device, model_type='bert',
        save_path=str(models_dir / 'distilbert_best.pth')
    )
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nLSTM Final Test Accuracy:       {lstm_history['test_acc'][-1]:.4f}")
    print(f"DistilBERT Final Test Accuracy: {bert_history['test_acc'][-1]:.4f}")
    print(f"\nWinner: {'DistilBERT' if bert_history['test_acc'][-1] > lstm_history['test_acc'][-1] else 'LSTM'}")
    
    return lstm_model, bert_model, lstm_history, bert_history


def predict_sentiment(text: str, model_type: str = 'distilbert', model_path: str = None) -> Tuple[str, float]:
    """
    Predict sentiment for a given text.
    
    Args:
        text: Input text
        model_type: 'lstm' or 'distilbert'
        model_path: Path to model checkpoint
        
    Returns:
        Tuple of (sentiment, confidence)
    """
    from typing import Tuple
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_path is None:
        model_path = Path(__file__).parent.parent / 'models' / f'{model_type}_best.pth'
    
    if model_type == 'distilbert':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        model = get_model('distilbert')
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        encoding = tokenizer(
            text, truncation=True, max_length=256,
            padding='max_length', return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(
                encoding['input_ids'].to(device),
                encoding['attention_mask'].to(device)
            )
            probs = torch.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1).item()
            confidence = probs[0][pred].item()
    else:
        raise NotImplementedError("LSTM inference requires vocabulary")
    
    sentiment = "Positive" if pred == 1 else "Negative"
    return sentiment, confidence


if __name__ == "__main__":
    lstm_model, bert_model, lstm_history, bert_history = main()
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (DistilBERT)")
    print("="*60)
    
    test_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Avoid at all costs.",
        "It was okay, not great but not terrible either.",
    ]
    
    for review in test_reviews:
        try:
            sentiment, confidence = predict_sentiment(review)
            print(f"\nReview: '{review[:50]}...'")
            print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")
        except Exception as e:
            print(f"Prediction error: {e}")
