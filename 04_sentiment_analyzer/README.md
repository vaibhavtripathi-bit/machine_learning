# Sentiment Analyzer - LSTM vs BERT

A sentiment analysis project comparing traditional LSTM architecture with modern transformer-based BERT on IMDB movie reviews.

## Features

- **LSTM Model**: Bidirectional LSTM with word embeddings
- **BERT Model**: Fine-tuned DistilBERT for faster training
- **Side-by-side Comparison**: Train and evaluate both approaches
- **Gradio UI**: Interactive demo interface (stretch goal)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train both models
python src/main.py
```

## Model Comparison

| Model | Test Accuracy | Training Speed | Parameters |
|-------|--------------|----------------|------------|
| LSTM | ~85% | Fast | ~3M |
| DistilBERT | ~92% | Slower | 66M |

## Project Structure

```
04_sentiment_analyzer/
├── data/                   # Dataset (auto-downloaded)
├── models/                 # Saved models
│   ├── lstm_best.pth
│   └── distilbert_best.pth
├── notebooks/              # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Data loading and preprocessing
│   ├── model.py            # LSTM and BERT models
│   ├── train.py            # Training loops
│   └── main.py             # Main pipeline
├── gradio_app/             # Gradio demo
├── requirements.txt
└── README.md
```

## Dataset

Uses the **IMDB Movie Reviews** dataset:
- 50,000 movie reviews (25k train, 25k test)
- Binary classification: Positive / Negative
- Average review length: ~230 words

## Architecture

### LSTM Model
```
Embedding(vocab_size, 128)
    ↓
Bidirectional LSTM(128→256, 2 layers)
    ↓
Dropout(0.5) → FC(512→256) → ReLU → Dropout(0.5) → FC(256→2)
```

### DistilBERT Model
```
DistilBERT-base-uncased (pretrained)
    ↓
[CLS] token embedding
    ↓
Dropout(0.3) → FC(768→384) → ReLU → Dropout(0.3) → FC(384→2)
```

## Key Concepts Learned

1. **Tokenization**: Word-level (LSTM) vs subword (BERT)
2. **Embeddings**: Learned embeddings vs pretrained contextual embeddings
3. **LSTM Architecture**: Understanding recurrence, hidden states, bidirectionality
4. **Transformers**: Self-attention mechanism and why it outperforms RNNs
5. **Fine-tuning**: Transfer learning with pretrained language models
6. **Sequence Padding**: Handling variable-length inputs

## Why BERT Wins

| Aspect | LSTM | BERT |
|--------|------|------|
| Context | Sequential, limited range | Full bidirectional attention |
| Pre-training | None (train from scratch) | Massive corpus (Wikipedia, Books) |
| Word Understanding | Static embeddings | Contextual embeddings |
| Long Dependencies | Difficult (vanishing gradients) | Native support |

## Usage

### Training
```python
from src.main import main
lstm_model, bert_model, lstm_history, bert_history = main()
```

### Inference
```python
from src.main import predict_sentiment

sentiment, confidence = predict_sentiment("This movie was amazing!")
print(f"{sentiment}: {confidence:.2%}")  # Positive: 95.23%
```

## Extending the Project

- Add Gradio UI for interactive demo
- Try other transformers (RoBERTa, ALBERT)
- Implement attention visualization
- Add multi-class sentiment (1-5 stars)
- Deploy as API endpoint

## License

MIT License
