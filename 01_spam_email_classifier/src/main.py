"""
Main script for training and evaluating the spam classifier.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import TextPreprocessor, load_and_prepare_data, download_dataset
from src.model import SpamClassifier
from src.evaluate import (
    calculate_metrics,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)


def main():
    """Main training pipeline."""
    print("="*60)
    print("SPAM EMAIL CLASSIFIER")
    print("="*60)
    
    data_path = Path(__file__).parent.parent / 'data' / 'spam.csv'
    if not data_path.exists():
        print("\nDownloading dataset...")
        download_dataset(str(data_path))
    
    print("\n1. Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data(str(data_path))
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print(f"   Spam ratio (train): {y_train.mean():.2%}")
    
    print("\n2. Preprocessing text with TF-IDF...")
    preprocessor = TextPreprocessor(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = preprocessor.fit_transform(X_train)
    X_test_tfidf = preprocessor.transform(X_test)
    print(f"   Feature dimensions: {X_train_tfidf.shape[1]}")
    
    print("\n3. Training Logistic Regression model...")
    model = SpamClassifier(model_type='logistic_regression')
    model.fit(X_train_tfidf, y_train)
    print("   Model trained successfully!")
    
    print("\n4. Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    
    metrics = calculate_metrics(y_test, y_pred, y_prob)
    print("\n   Performance Metrics:")
    print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall:    {metrics['recall']:.4f}")
    print(f"   - F1 Score:  {metrics['f1_score']:.4f}")
    print(f"   - ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print_classification_report(y_test, y_pred)
    
    print("\n5. Top spam indicators:")
    feature_names = preprocessor.get_feature_names()
    importance = model.get_feature_importance(feature_names, top_n=10)
    for feature, score in importance.items():
        indicator = "SPAM" if score > 0 else "HAM"
        print(f"   '{feature}': {score:.4f} ({indicator})")
    
    print("\n6. Saving model...")
    model_path = Path(__file__).parent.parent / 'models' / 'spam_classifier.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    
    import pickle
    preprocessor_path = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"   Preprocessor saved to {preprocessor_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    return model, preprocessor, metrics


def predict_message(message: str, model_path: str = None, preprocessor_path: str = None):
    """
    Predict if a message is spam or ham.
    
    Args:
        message: Text message to classify
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        
    Returns:
        Tuple of (prediction, probability)
    """
    import pickle
    
    if model_path is None:
        model_path = Path(__file__).parent.parent / 'models' / 'spam_classifier.pkl'
    if preprocessor_path is None:
        preprocessor_path = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'
    
    model = SpamClassifier.load(str(model_path))
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    import pandas as pd
    message_series = pd.Series([message])
    features = preprocessor.transform(message_series)
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    label = "SPAM" if prediction == 1 else "HAM"
    return label, probability


if __name__ == "__main__":
    model, preprocessor, metrics = main()
    
    print("\n" + "="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    test_messages = [
        "Congratulations! You've won a $1000 gift card. Call now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Click here immediately!",
        "Can you pick up some milk on your way home?",
        "FREE entry to win a brand new iPhone! Text WIN to 12345",
    ]
    
    for msg in test_messages:
        label, prob = predict_message(msg)
        print(f"\nMessage: '{msg[:50]}...'")
        print(f"Prediction: {label} (confidence: {prob:.2%})")
