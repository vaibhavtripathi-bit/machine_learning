"""
Evaluation module for house price prediction.
Provides metrics calculation and visualization.
"""

from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in a formatted way."""
    print("\n" + "="*50)
    print("REGRESSION METRICS")
    print("="*50)
    print(f"  RMSE:  ${metrics['rmse']:,.2f}")
    print(f"  MAE:   ${metrics['mae']:,.2f}")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print("="*50)


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot actual vs predicted prices.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='none')
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0].set_xlabel('Actual Price ($)')
    axes[0].set_ylabel('Predicted Price ($)')
    axes[0].set_title('Actual vs Predicted House Prices')
    axes[0].grid(True, alpha=0.3)
    
    r2 = r2_score(y_true, y_pred)
    axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes, 
                  fontsize=12, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Price ($)')
    axes[1].set_ylabel('Residuals ($)')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of residuals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residuals ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Residuals')
    axes[0].grid(True, alpha=0.3)
    
    percentage_errors = (residuals / y_true) * 100
    axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Percentage Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Percentage Errors')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_feature_importance(
    importance_dict: Dict[str, float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importances.
    
    Args:
        importance_dict: Dictionary of feature importances
        save_path: Path to save the plot
    """
    features = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    plt.figure(figsize=(10, 8))
    plt.barh(features[::-1], values[::-1], color='steelblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_model_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Plot model comparison results.
    
    Args:
        results: Dictionary of model CV results
        save_path: Path to save the plot
    """
    models = list(results.keys())
    rmse_means = [results[m]['rmse_mean'] for m in models]
    rmse_stds = [results[m]['rmse_std'] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, rmse_means, yerr=rmse_stds, capsize=5, 
                   color='steelblue', edgecolor='black')
    plt.ylabel('RMSE ($)')
    plt.title('Model Comparison (Cross-Validation RMSE)')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, mean in zip(bars, rmse_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'${mean:,.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()
