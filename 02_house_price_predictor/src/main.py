"""
Main script for training and evaluating house price predictor.
"""

import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import HousingDataPreprocessor, download_dataset, load_and_prepare_data
from src.model import HousePricePredictor, compare_models
from src.evaluate import calculate_metrics, print_metrics, plot_actual_vs_predicted, plot_feature_importance


def main():
    """Main training pipeline."""
    print("="*60)
    print("HOUSE PRICE PREDICTOR")
    print("="*60)
    
    data_path = Path(__file__).parent.parent / 'data' / 'ames_housing.csv'
    if not data_path.exists():
        print("\nDownloading dataset...")
        download_dataset(str(data_path.parent))
    
    print("\n1. Loading and preparing data...")
    X_train_df, X_test_df, y_train, y_test = load_and_prepare_data(str(data_path))
    print(f"   Training samples: {len(X_train_df)}")
    print(f"   Testing samples: {len(X_test_df)}")
    print(f"   Features: {X_train_df.shape[1]}")
    print(f"   Price range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    
    print("\n2. Preprocessing and feature engineering...")
    preprocessor = HousingDataPreprocessor()
    X_train, y_train_arr = preprocessor.fit_transform(
        X_train_df.assign(SalePrice=y_train),
        target_col='SalePrice'
    )
    X_test = preprocessor.transform(X_test_df)
    y_test_arr = y_test.values
    
    print(f"   Final feature dimensions: {X_train.shape[1]}")
    
    print("\n3. Comparing models with cross-validation...")
    cv_results = compare_models(X_train, y_train_arr, cv=5)
    
    best_model_type = min(cv_results, key=lambda x: cv_results[x]['rmse_mean'])
    print(f"\n   Best model: {best_model_type}")
    
    print(f"\n4. Training final {best_model_type} model...")
    model = HousePricePredictor(model_type=best_model_type)
    model.fit(X_train, y_train_arr)
    print("   Model trained successfully!")
    
    print("\n5. Evaluating on test set...")
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test_arr, y_pred)
    print_metrics(metrics)
    
    print("\n6. Top important features:")
    feature_names = preprocessor.get_feature_names()
    importance = model.get_feature_importance(feature_names, top_n=10)
    for i, (feature, score) in enumerate(importance.items(), 1):
        print(f"   {i:2d}. {feature}: {score:.4f}")
    
    print("\n7. Saving model and preprocessor...")
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(str(model_dir / 'house_price_model.pkl'))
    
    with open(model_dir / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"   Preprocessor saved to {model_dir / 'preprocessor.pkl'}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    return model, preprocessor, metrics


def predict_price(features_dict: dict, model_path: str = None, preprocessor_path: str = None):
    """
    Predict house price for given features.
    
    Args:
        features_dict: Dictionary of house features
        model_path: Path to saved model
        preprocessor_path: Path to saved preprocessor
        
    Returns:
        Predicted price
    """
    import pandas as pd
    
    if model_path is None:
        model_path = Path(__file__).parent.parent / 'models' / 'house_price_model.pkl'
    if preprocessor_path is None:
        preprocessor_path = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'
    
    model = HousePricePredictor.load(str(model_path))
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    df = pd.DataFrame([features_dict])
    X = preprocessor.transform(df)
    
    prediction = model.predict(X)[0]
    return prediction


if __name__ == "__main__":
    model, preprocessor, metrics = main()
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    sample_house = {
        'GrLivArea': 1500,
        'OverallQual': 7,
        'OverallCond': 5,
        'YearBuilt': 2000,
        'YrSold': 2010,
        'TotalBsmtSF': 1000,
        'FullBath': 2,
        'HalfBath': 1,
        'BedroomAbvGr': 3,
        'GarageCars': 2,
        'GarageArea': 500,
    }
    
    print("\nSample house features:")
    for key, value in sample_house.items():
        print(f"   {key}: {value}")
    
    print(f"\nNote: Full prediction requires all features from the dataset.")
    print(f"Model expects {len(preprocessor.get_feature_names())} features.")
