"""
Model module for house price prediction.
Implements Linear Regression, Random Forest, and XGBoost.
"""

import pickle
from typing import Dict, Optional
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class HousePricePredictor:
    """House price predictor supporting multiple regression algorithms."""
    
    MODELS = {
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
    }
    
    if XGBOOST_AVAILABLE:
        MODELS['xgboost'] = XGBRegressor
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to use
            **kwargs: Additional model parameters
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(self.MODELS.keys())}")
        
        self.model_type = model_type
        
        default_params = {
            'ridge': {'alpha': 1.0, 'random_state': 42},
            'lasso': {'alpha': 0.001, 'random_state': 42},
            'elastic_net': {'alpha': 0.001, 'l1_ratio': 0.5, 'random_state': 42},
            'random_forest': {'n_estimators': 100, 'max_depth': 15, 'random_state': 42, 'n_jobs': -1},
            'gradient_boosting': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        }
        
        params = {**default_params.get(model_type, {}), **kwargs}
        self.model = self.MODELS[model_type](**params)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HousePricePredictor':
        """
        Fit the model on training data.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            self
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict house prices.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted prices
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            cv: Number of folds
            
        Returns:
            Dictionary with CV scores
        """
        neg_mse_scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        r2_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        
        return {
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std()
        }
    
    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importances.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return {}
        
        top_indices = np.argsort(importance)[-top_n:][::-1]
        return {feature_names[i]: float(importance[i]) for i in top_indices}
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'HousePricePredictor':
        """Load model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def compare_models(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> Dict[str, Dict]:
    """
    Compare multiple models using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of CV folds
        
    Returns:
        Dictionary of model results
    """
    models_to_compare = ['ridge', 'random_forest', 'gradient_boosting']
    if XGBOOST_AVAILABLE:
        models_to_compare.append('xgboost')
    
    results = {}
    
    for model_type in models_to_compare:
        print(f"  Evaluating {model_type}...")
        model = HousePricePredictor(model_type=model_type)
        cv_results = model.cross_validate(X_train, y_train, cv=cv)
        results[model_type] = cv_results
        print(f"    RMSE: ${cv_results['rmse_mean']:,.0f} (+/- ${cv_results['rmse_std']:,.0f})")
    
    return results
