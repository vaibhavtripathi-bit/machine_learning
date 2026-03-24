"""
Training pipeline with MLflow tracking.
"""

import os
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


def load_data():
    """Load and prepare training data."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y, iris.target_names


def train_model(X_train, y_train, params: dict):
    """Train a model with given parameters."""
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
    }
    
    return metrics


def run_experiment(experiment_name: str = "iris_classification"):
    """Run a training experiment with MLflow tracking."""
    print("="*60)
    print("TRAINING PIPELINE WITH MLFLOW")
    print("="*60)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    
    print("\n1. Loading data...")
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    param_grid = [
        {'n_estimators': 50, 'max_depth': 3, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        {'n_estimators': 200, 'max_depth': 10, 'random_state': 42},
    ]
    
    print("\n2. Running experiments...")
    best_run = None
    best_accuracy = 0
    
    for i, params in enumerate(param_grid):
        with mlflow.start_run(run_name=f"run_{i+1}"):
            print(f"\n   Experiment {i+1}/{len(param_grid)}")
            print(f"   Parameters: {params}")
            
            mlflow.log_params(params)
            
            model = train_model(X_train, y_train, params)
            
            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1 Score: {metrics['f1']:.4f}")
            
            mlflow.sklearn.log_model(model, "model")
            
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_run = mlflow.active_run().info.run_id
    
    print(f"\n3. Best run: {best_run}")
    print(f"   Best accuracy: {best_accuracy:.4f}")
    
    print("\n4. Registering best model...")
    model_uri = f"runs:/{best_run}/model"
    model_name = "iris_classifier"
    
    try:
        result = mlflow.register_model(model_uri, model_name)
        print(f"   Model registered: {model_name} (version {result.version})")
    except Exception as e:
        print(f"   Model registration skipped: {e}")
    
    print("\n5. Saving model locally...")
    output_dir = Path(__file__).parent.parent / "serving" / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(best_model, output_dir / "model.joblib")
    
    with open(output_dir / "metadata.json", "w") as f:
        import json
        json.dump({
            "model_name": model_name,
            "run_id": best_run,
            "accuracy": best_accuracy,
            "target_names": list(target_names)
        }, f, indent=2)
    
    print(f"   Model saved to {output_dir}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nView experiments: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    
    return best_run, best_accuracy


if __name__ == "__main__":
    run_experiment()
