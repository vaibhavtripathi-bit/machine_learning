"""
Flask API for spam classification.
Provides REST endpoint for spam/ham prediction.
"""

import os
import sys
from pathlib import Path
import pickle

from flask import Flask, request, jsonify

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SpamClassifier

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent.parent / 'models' / 'spam_classifier.pkl'
PREPROCESSOR_PATH = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'

model = None
preprocessor = None


def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor
    
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Run 'python src/main.py' first to train the model."
        )
    
    model = SpamClassifier.load(str(MODEL_PATH))
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    
    print("Model loaded successfully!")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if a message is spam or ham.
    
    Request body:
        {"message": "Your text message here"}
    
    Response:
        {
            "message": "original message",
            "prediction": "SPAM" or "HAM",
            "confidence": 0.95,
            "is_spam": true/false
        }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field in request body'}), 400
    
    message = data['message']
    
    if not isinstance(message, str) or len(message.strip()) == 0:
        return jsonify({'error': 'Message must be a non-empty string'}), 400
    
    import pandas as pd
    message_series = pd.Series([message])
    features = preprocessor.transform(message_series)
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    label = "SPAM" if prediction == 1 else "HAM"
    
    return jsonify({
        'message': message,
        'prediction': label,
        'confidence': round(float(probability if prediction == 1 else 1 - probability), 4),
        'spam_probability': round(float(probability), 4),
        'is_spam': bool(prediction == 1)
    })


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict spam/ham for multiple messages.
    
    Request body:
        {"messages": ["message1", "message2", ...]}
    
    Response:
        {"predictions": [...]}
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    
    if not data or 'messages' not in data:
        return jsonify({'error': 'Missing "messages" field in request body'}), 400
    
    messages = data['messages']
    
    if not isinstance(messages, list) or len(messages) == 0:
        return jsonify({'error': 'Messages must be a non-empty list'}), 400
    
    import pandas as pd
    messages_series = pd.Series(messages)
    features = preprocessor.transform(messages_series)
    
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    
    results = []
    for msg, pred, prob in zip(messages, predictions, probabilities):
        label = "SPAM" if pred == 1 else "HAM"
        results.append({
            'message': msg,
            'prediction': label,
            'spam_probability': round(float(prob), 4),
            'is_spam': bool(pred == 1)
        })
    
    return jsonify({'predictions': results})


if __name__ == '__main__':
    load_model()
    print("\nStarting Flask API server...")
    print("Endpoints:")
    print("  GET  /health  - Health check")
    print("  POST /predict - Classify single message")
    print("  POST /predict/batch - Classify multiple messages")
    print("\nExample:")
    print('  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d \'{"message": "You won a free iPhone!"}\'')
    print()
    app.run(host='0.0.0.0', port=5000, debug=True)
