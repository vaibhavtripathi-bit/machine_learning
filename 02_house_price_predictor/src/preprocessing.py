"""
Data preprocessing module for house price prediction.
Handles missing values, feature engineering, and encoding.
"""

import os
import urllib.request
from typing import Tuple, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class HousingDataPreprocessor:
    """Preprocessor for housing data with feature engineering."""
    
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_columns = []
        self.categorical_columns = []
        self.is_fitted = False
        
    def _identify_columns(self, df: pd.DataFrame) -> None:
        """Identify numerical and categorical columns."""
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if 'SalePrice' in self.numerical_columns:
            self.numerical_columns.remove('SalePrice')
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        df = df.copy()
        
        if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
            df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)
        
        if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
            df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        
        if 'YearRemodAdd' in df.columns and 'YrSold' in df.columns:
            df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        
        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        available_bath = [c for c in bath_cols if c in df.columns]
        if available_bath:
            df['TotalBath'] = sum(df[c].fillna(0) for c in available_bath)
        
        if 'GarageArea' in df.columns and 'GarageCars' in df.columns:
            df['GarageScore'] = df['GarageArea'].fillna(0) * df['GarageCars'].fillna(0)
        
        if 'OverallQual' in df.columns and 'OverallCond' in df.columns:
            df['OverallScore'] = df['OverallQual'] * df['OverallCond']
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'SalePrice') -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (features, target)
        """
        df = self._engineer_features(df)
        
        y = df[target_col].values if target_col in df.columns else None
        df = df.drop(columns=[target_col], errors='ignore')
        
        self._identify_columns(df)
        
        if self.numerical_columns:
            df[self.numerical_columns] = self.numerical_imputer.fit_transform(df[self.numerical_columns])
        
        if self.categorical_columns:
            df[self.categorical_columns] = self.categorical_imputer.fit_transform(df[self.categorical_columns])
        
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        feature_cols = self.numerical_columns + self.categorical_columns
        X = df[feature_cols].values
        
        X = self.scaler.fit_transform(X)
        
        self.feature_names = feature_cols
        self.is_fitted = True
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df = self._engineer_features(df)
        df = df.drop(columns=['SalePrice'], errors='ignore')
        
        if self.numerical_columns:
            df[self.numerical_columns] = self.numerical_imputer.transform(df[self.numerical_columns])
        
        if self.categorical_columns:
            df[self.categorical_columns] = self.categorical_imputer.transform(df[self.categorical_columns])
        
        for col in self.categorical_columns:
            df[col] = df[col].astype(str).apply(
                lambda x: self.label_encoders[col].transform([x])[0] 
                if x in self.label_encoders[col].classes_ 
                else -1
            )
        
        feature_cols = self.numerical_columns + self.categorical_columns
        X = df[feature_cols].values
        X = self.scaler.transform(X)
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self.feature_names


def download_dataset(save_dir: str = 'data') -> str:
    """
    Download the Ames Housing dataset.
    
    Args:
        save_dir: Directory to save the dataset
        
    Returns:
        Path to the saved dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    
    url = "https://raw.githubusercontent.com/selva86/datasets/master/AmesHousing.csv"
    save_path = os.path.join(save_dir, 'ames_housing.csv')
    
    print("Downloading Ames Housing dataset...")
    urllib.request.urlretrieve(url, save_path)
    
    df = pd.read_csv(save_path)
    print(f"Dataset saved to {save_path}")
    print(f"Shape: {df.shape}")
    
    return save_path


def load_and_prepare_data(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and split the housing data.
    
    Args:
        data_path: Path to the CSV file
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test DataFrames
    """
    df = pd.read_csv(data_path)
    
    if 'Order' in df.columns:
        df = df.drop(columns=['Order'])
    if 'PID' in df.columns:
        df = df.drop(columns=['PID'])
    
    df = df.dropna(subset=['SalePrice'])
    
    df = df[df['SalePrice'] > 0]
    df = df[df['SalePrice'] < df['SalePrice'].quantile(0.99)]
    
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
