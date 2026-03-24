"""
Text preprocessing module for spam classification.
Handles text cleaning, tokenization, and TF-IDF vectorization.
"""

import re
import string
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class TextPreprocessor:
    """Preprocessor for text data with TF-IDF vectorization."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the preprocessor.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to extract
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer: Optional[TfidfVectorizer] = None
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """
        Fit the vectorizer and transform texts.
        
        Args:
            texts: Series of text documents
            
        Returns:
            TF-IDF feature matrix
        """
        cleaned_texts = texts.apply(self.clean_text)
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True
        )
        
        return self.vectorizer.fit_transform(cleaned_texts).toarray()
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: Series of text documents
            
        Returns:
            TF-IDF feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        cleaned_texts = texts.apply(self.clean_text)
        return self.vectorizer.transform(cleaned_texts).toarray()
    
    def get_feature_names(self):
        """Get feature names from the vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        return self.vectorizer.get_feature_names_out()


def load_and_prepare_data(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load SMS spam dataset and prepare train/test splits.
    
    Args:
        data_path: Path to the CSV file
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(data_path, encoding='latin-1')
    
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    elif 'label' not in df.columns:
        df.columns = ['label', 'message'] + list(df.columns[2:])
        df = df[['label', 'message']]
    
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'],
        df['label'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    
    return X_train, X_test, y_train, y_test


def download_dataset(save_path: str = 'data/spam.csv') -> str:
    """
    Download the SMS Spam Collection dataset.
    
    Args:
        save_path: Path to save the dataset
        
    Returns:
        Path to the saved dataset
    """
    import os
    import urllib.request
    import zipfile
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = save_path.replace('.csv', '.zip')
    
    print("Downloading SMS Spam Collection dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(save_path))
    
    tsv_path = os.path.join(os.path.dirname(save_path), 'SMSSpamCollection')
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['label', 'message'])
    df.to_csv(save_path, index=False)
    
    os.remove(zip_path)
    if os.path.exists(tsv_path):
        os.remove(tsv_path)
    readme_path = os.path.join(os.path.dirname(save_path), 'readme')
    if os.path.exists(readme_path):
        os.remove(readme_path)
    
    print(f"Dataset saved to {save_path}")
    print(f"Total samples: {len(df)}")
    print(f"Spam: {(df['label'] == 'spam').sum()}, Ham: {(df['label'] == 'ham').sum()}")
    
    return save_path
