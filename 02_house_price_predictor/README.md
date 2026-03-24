# House Price Predictor

A regression model that predicts house prices using the Ames Housing dataset with feature engineering and XGBoost.

## Features

- **Feature Engineering**: Creates derived features (TotalSF, HouseAge, TotalBath, etc.)
- **Multiple Models**: Ridge, Random Forest, Gradient Boosting, XGBoost
- **Model Comparison**: Cross-validation to select the best model
- **Visualization**: Actual vs Predicted plots, residual analysis, feature importance

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python src/main.py
```

## Dataset

Uses the [Ames Housing Dataset](http://jse.amstat.org/v19n3/decock.pdf):
- 2,930 residential properties in Ames, Iowa
- 79 explanatory variables (features)
- Target: Sale price of the house

The dataset is automatically downloaded on first run.

## Model Performance

Using XGBoost with cross-validation:

| Metric | Score |
|--------|-------|
| RMSE | ~$25,000 |
| MAE | ~$16,000 |
| R² | ~0.91 |
| MAPE | ~8% |

## Project Structure

```
02_house_price_predictor/
├── data/                   # Dataset (auto-downloaded)
│   └── ames_housing.csv
├── models/                 # Trained models
│   ├── house_price_model.pkl
│   └── preprocessor.pkl
├── notebooks/              # Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Feature engineering, encoding
│   ├── model.py            # Regression models
│   ├── evaluate.py         # Metrics and visualization
│   └── main.py             # Training pipeline
├── dashboard/              # Visualization dashboard
├── requirements.txt
└── README.md
```

## Feature Engineering

The preprocessor creates several derived features:

| Feature | Description |
|---------|-------------|
| TotalSF | Total square footage (GrLivArea + TotalBsmtSF) |
| HouseAge | Age of house at sale (YrSold - YearBuilt) |
| RemodAge | Years since remodel |
| TotalBath | Total bathrooms (full + half) |
| GarageScore | Garage area × cars capacity |
| OverallScore | Quality × Condition |

## Models Compared

1. **Ridge Regression**: Linear model with L2 regularization
2. **Random Forest**: Ensemble of decision trees
3. **Gradient Boosting**: Sequential tree ensemble
4. **XGBoost**: Optimized gradient boosting (usually best)

## Key Concepts Learned

1. **Feature Engineering**: Creating meaningful features from raw data
2. **One-Hot Encoding**: Converting categorical variables
3. **Cross-Validation**: Robust model evaluation
4. **XGBoost**: Understanding gradient boosting
5. **RMSE vs MAE**: Different error metrics for different use cases
6. **Feature Importance**: Understanding what drives predictions

## Top Important Features

Typical top features for house price prediction:
1. OverallQual (Overall quality rating)
2. GrLivArea (Above ground living area)
3. TotalSF (Total square footage)
4. GarageCars (Garage car capacity)
5. YearBuilt (Year house was built)

## Usage Example

```python
from src.main import predict_price

# Predict price for a house
features = {
    'GrLivArea': 1500,
    'OverallQual': 7,
    'YearBuilt': 2000,
    'TotalBsmtSF': 1000,
    'GarageCars': 2,
    # ... more features
}

price = predict_price(features)
print(f"Predicted price: ${price:,.2f}")
```

## Extending the Project

- Add more advanced feature engineering
- Implement stacking/blending ensemble
- Build a Streamlit dashboard
- Deploy as a web service
- Add confidence intervals for predictions

## License

MIT License
