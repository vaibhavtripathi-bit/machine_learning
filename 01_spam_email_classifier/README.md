# Spam Email Classifier

A binary text classifier that detects spam messages using TF-IDF vectorization and Logistic Regression.

## Features

- **TF-IDF Vectorization**: Converts text to numerical features with n-grams
- **Logistic Regression**: Fast, interpretable classification model
- **Feature Analysis**: View top spam/ham indicators
- **REST API**: Flask endpoint for real-time predictions
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/main.py

# Start the API server
python api/app.py
```

## Dataset

Uses the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) dataset:
- 5,574 SMS messages
- 747 spam, 4,827 ham (legitimate)
- Binary classification task

The dataset is automatically downloaded on first run.

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~97.5% |
| Precision | ~100% |
| Recall | ~85% |
| F1 Score | ~92% |
| ROC-AUC | ~99% |

## Project Structure

```
01_spam_email_classifier/
├── data/                   # Dataset (auto-downloaded)
│   └── spam.csv
├── models/                 # Trained models
│   ├── spam_classifier.pkl
│   └── preprocessor.pkl
├── notebooks/              # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # TF-IDF vectorization
│   ├── model.py            # Classifier implementation
│   ├── evaluate.py         # Metrics and visualization
│   └── main.py             # Training pipeline
├── api/
│   └── app.py              # Flask REST API
├── requirements.txt
└── README.md
```

## API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won a free iPhone!"}'
```

Response:
```json
{
  "message": "Congratulations! You won a free iPhone!",
  "prediction": "SPAM",
  "confidence": 0.9823,
  "spam_probability": 0.9823,
  "is_spam": true
}
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Free gift!", "Meeting at 3pm?"]}'
```

## Key Concepts Learned

1. **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency for text representation
2. **Train/Test Split**: Proper data splitting with stratification
3. **Confusion Matrix**: Understanding True/False Positives/Negatives
4. **Precision vs Recall Trade-off**: High precision to avoid false spam flags
5. **Model Serialization**: Saving/loading models with pickle

## Top Spam Indicators

Words with highest positive coefficients (spam indicators):
- "free", "win", "prize", "call", "txt", "claim", "urgent"

Words with highest negative coefficients (ham indicators):
- "will", "can", "are", "your", "have"

## Extending the Project

- Add Naive Bayes comparison
- Implement word embeddings (Word2Vec)
- Add email-specific features (headers, links)
- Deploy to cloud (AWS Lambda, GCP Cloud Functions)

## License

MIT License
