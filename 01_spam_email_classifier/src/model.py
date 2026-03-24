"""
Model module for spam classification.
Implements Logistic Regression and Naive Bayes classifiers.
"""

import pickle
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


class SpamClassifier:
    """Spam classifier supporting multiple algorithms."""
    
    MODELS = {
        'logistic_regression': LogisticRegression,
        'naive_bayes': MultinomialNB,
        'random_forest': RandomForestClassifier
    }
    
    def __init__(self, model_type: str = 'logistic_regression', **kwargs):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model ('logistic_regression', 'naive_bayes', 'random_forest')
            **kwargs: Additional arguments for the model
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(self.MODELS.keys())}")
        
        self.model_type = model_type
        
        default_params = {
            'logistic_regression': {'max_iter': 1000, 'random_state': 42},
            'naive_bayes': {'alpha': 1.0},
            'random_forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        }
        
        params = {**default_params.get(model_type, {}), **kwargs}
        self.model = self.MODELS[model_type](**params)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SpamClassifier':
        """
        Fit the model on training data.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            self
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: np.ndarray, top_n: int = 20) -> Dict[str, float]:
        """
        Get top feature importances (for logistic regression).
        
        Args:
            feature_names: Array of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if self.model_type == 'logistic_regression':
            coef = self.model.coef_[0]
            top_indices = np.argsort(np.abs(coef))[-top_n:][::-1]
            return {feature_names[i]: coef[i] for i in top_indices}
        elif self.model_type == 'random_forest':
            importance = self.model.feature_importances_
            top_indices = np.argsort(importance)[-top_n:][::-1]
            return {feature_names[i]: importance[i] for i in top_indices}
        else:
            return {}
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SpamClassifier':
        """Load a model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
