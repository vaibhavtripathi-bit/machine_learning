# End-to-End Flow: House Price Predictor

## Overview

This document explains the complete pipeline from raw housing data to a trained regression model that predicts sale prices.

---

## Flow Diagram

```
Ames Housing Dataset (CSV)
          │
          ▼
┌──────────────────────┐
│   Data Loading       │  ← 2,930 houses, 79 features
│   & Cleaning         │    Remove outliers, drop ID cols
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Feature Engineering │  ← Create TotalSF, HouseAge,
│                      │    TotalBath, GarageScore...
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Handle Missing      │  ← Numerical → median impute
│  Values              │    Categorical → mode impute
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Encode Categorical  │  ← Label encoding for tree models
│  Features            │    (e.g. Neighborhood, HouseStyle)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Feature Scaling     │  ← StandardScaler: mean=0, std=1
│                      │
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
  X_train      X_test
     │
     ▼
┌──────────────────────┐
│  Model Comparison    │  ← Ridge, Random Forest,
│  (5-fold CV)         │    Gradient Boosting, XGBoost
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Best Model          │  ← XGBoost (lowest RMSE)
│  Full Training       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Evaluation          │  ← RMSE, MAE, R², MAPE
│  on Test Set         │    Actual vs Predicted plot
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Model + Preprocessor│  ← house_price_model.pkl
│  Saved               │    preprocessor.pkl
└──────────────────────┘
```

---

## Step-by-Step Breakdown

### Step 1: Data Loading & Cleaning
**File**: `src/preprocessing.py` → `load_and_prepare_data()`

- Loads the [Ames Housing Dataset](http://jse.amstat.org/v19n3/decock.pdf)
- 2,930 houses sold in Ames, Iowa (2006–2010)
- Removes:
  - ID columns (`Order`, `PID`) — no predictive value
  - Rows with missing `SalePrice`
  - Outliers: top 1% prices (unusually expensive properties)

```python
X_train, X_test, y_train, y_test = load_and_prepare_data('data/ames_housing.csv')
# Target: SalePrice (continuous, in USD)
```

---

### Step 2: Feature Engineering
**File**: `src/preprocessing.py` → `HousingDataPreprocessor._engineer_features()`

Raw features alone don't tell the full story. New features are created:

| New Feature | Formula | Why It Matters |
|-------------|---------|----------------|
| `TotalSF` | `GrLivArea + TotalBsmtSF` | Total living space is a strong price driver |
| `HouseAge` | `YrSold - YearBuilt` | Older houses generally sell for less |
| `RemodAge` | `YrSold - YearRemodAdd` | Recent renovations boost price |
| `TotalBath` | `FullBath + HalfBath + BsmtFullBath + BsmtHalfBath` | More bathrooms = higher price |
| `GarageScore` | `GarageArea × GarageCars` | Garage quality composite |
| `OverallScore` | `OverallQual × OverallCond` | Quality × condition interaction |

---

### Step 3: Handle Missing Values
**File**: `src/preprocessing.py`

Different strategies for different column types:

```
Numerical columns   →  Median imputation (robust to outliers)
Categorical columns →  Most-frequent (mode) imputation
```

Why median over mean for numerical? Outliers in prices/areas skew the mean badly. The median is robust.

---

### Step 4: Categorical Encoding
**File**: `src/preprocessing.py`

- **Label Encoding**: Each unique category gets an integer
- Works well for tree-based models (XGBoost, Random Forest)
- For linear models, one-hot encoding would be better

Example:
```
Neighborhood: CollgCr → 0, Veenker → 1, Crawfor → 2 ...
```

---

### Step 5: Feature Scaling
**File**: `src/preprocessing.py`

`StandardScaler` transforms each feature:
```
z = (x - mean) / std
```

Why scale? Without scaling, features with large values (e.g. `GrLivArea` in sq ft) dominate over small-valued features (e.g. `FullBath` = 1-4).

---

### Step 6: Model Comparison
**File**: `src/model.py` → `compare_models()`

All models are evaluated using **5-Fold Cross-Validation** to get honest performance estimates:

```
Full Training Set split into 5 folds:
  Fold 1: [████ ████ ████ ████ ░░░░] → train on 4, test on 1
  Fold 2: [████ ████ ████ ░░░░ ████] → train on 4, test on 1
  ...
  Average RMSE across 5 folds = CV score
```

| Model | CV RMSE | Strengths |
|-------|---------|-----------|
| Ridge | ~$35k | Fast, interpretable |
| Random Forest | ~$28k | Handles non-linearity |
| Gradient Boosting | ~$26k | Good generalisation |
| **XGBoost** | **~$25k** | **Best accuracy, handles missing data** |

---

### Step 7: XGBoost Training
**File**: `src/model.py` → `HousePricePredictor`

XGBoost builds **gradient-boosted decision trees** sequentially:

```
Prediction = Tree₁ + Tree₂ + Tree₃ + ... + TreeN
              ↑
         Each tree corrects residuals from previous trees
```

Key hyperparameters used:
```python
n_estimators = 200    # Number of trees
max_depth = 6         # Max tree depth (controls overfitting)
learning_rate = 0.1   # How much each tree contributes
```

---

### Step 8: Evaluation
**File**: `src/evaluate.py`

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | `√(mean((y - ŷ)²))` | Average prediction error in $ |
| **MAE** | `mean(|y - ŷ|)` | Average absolute error (robust to outliers) |
| **R²** | `1 - SS_res/SS_tot` | 0.91 means model explains 91% of variance |
| **MAPE** | `mean(|y-ŷ|/y)×100` | Error as % of actual price |

**Residual plot** checks assumptions:
- Points scattered randomly → good model
- Patterns → model is missing something

---

### Step 9: Feature Importance

After training, XGBoost assigns importance scores:

```
OverallQual   ████████████████████  (most important)
TotalSF       ████████████████
GrLivArea     ████████████
GarageCars    ████████
YearBuilt     ██████
...
```

This tells you **what actually drives house prices** — a key insight for real estate.

---

## Data Flow Through Code

```
ames_housing.csv
  └── load_and_prepare_data()
        ├── X_train (DataFrame) ─► HousingDataPreprocessor.fit_transform()
        │     + y_train               ├── _engineer_features()   → new columns
        │                             ├── numerical_imputer      → fill NaN
        │                             ├── categorical_imputer    → fill NaN
        │                             ├── label_encoders         → int categories
        │                             └── scaler                 → normalized
        │                                    └── X_train_processed (np.ndarray)
        │                                           └── HousePricePredictor.fit()
        │                                                  └── trained XGBoost
        │
        └── X_test (DataFrame) ─► HousingDataPreprocessor.transform()
                                        └── X_test_processed (np.ndarray)
                                               └── HousePricePredictor.predict()
                                                      └── y_pred → metrics
```

---

## Running the Full Flow

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train (downloads data, engineers features, trains, evaluates)
python src/main.py

# Expected output:
# 1. Loading data...        → 2344 train, 586 test samples
# 2. Preprocessing...       → 85 final features
# 3. Comparing models...    → Ridge, RF, GBM, XGBoost CV scores
# 4. Training best model... → XGBoost
# 5. Evaluation...          → RMSE: ~$25k, R²: ~0.91
# 6. Feature importance     → Top 10 features
# 7. Model saved            → models/house_price_model.pkl
```

---

## Key Learnings

| Concept | Where Used |
|---------|-----------|
| Feature Engineering | Creating `TotalSF`, `HouseAge` from raw data |
| Imputation strategies | Median for numerical, mode for categorical |
| Cross-validation | Honest model comparison without touching test set |
| XGBoost | Gradient boosting outperforms linear models on tabular data |
| RMSE vs MAE | RMSE penalizes large errors more (good for house prices) |
| Feature Importance | Understanding what drives predictions |
