# End-to-End Flow: Spam Email Classifier

## Overview

This document explains the complete pipeline from raw text data to a deployed spam detection API.

---

## Flow Diagram

```
Raw SMS Text
     │
     ▼
┌─────────────────┐
│  Data Loading   │  ← SMS Spam Collection dataset (5,574 messages)
│  (spam.csv)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Cleaning   │  ← Lowercase, remove URLs, digits, punctuation
│ preprocessing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TF-IDF         │  ← Convert text → numerical feature matrix
│  Vectorization  │    max_features=5000, ngram_range=(1,2)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│     Train / Test Split      │  ← 80% train, 20% test (stratified)
└──────────┬──────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  X_train     X_test
     │
     ▼
┌─────────────────┐
│ Logistic        │  ← Trains on TF-IDF features
│ Regression      │    max_iter=1000
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │  ← Accuracy, Precision, Recall, F1, ROC-AUC
│   on Test Set   │    Confusion Matrix, Precision-Recall curve
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Saved    │  ← spam_classifier.pkl + preprocessor.pkl
│  to Disk        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flask REST     │  ← POST /predict → { prediction, confidence }
│  API (port 5000)│    POST /predict/batch → bulk predictions
└─────────────────┘
```

---

## Step-by-Step Breakdown

### Step 1: Data Loading
**File**: `src/preprocessing.py` → `download_dataset()`, `load_and_prepare_data()`

- Downloads the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- 5,574 messages: 747 spam (13.4%), 4,827 ham (86.6%)
- Maps labels: `ham → 0`, `spam → 1`
- Splits into train/test with stratification to preserve class ratio

```python
X_train, X_test, y_train, y_test = load_and_prepare_data('data/spam.csv')
# X: pd.Series of raw text messages
# y: pd.Series of 0/1 labels
```

---

### Step 2: Text Preprocessing
**File**: `src/preprocessing.py` → `TextPreprocessor.clean_text()`

Each message goes through:
1. **Lowercase**: `"FREE PRIZE" → "free prize"`
2. **Remove URLs**: strips `http://...`
3. **Remove digits**: `"call 0800123"` → `"call "`
4. **Remove punctuation**: `"win!!!"` → `"win"`
5. **Strip whitespace**: normalizes spacing

```python
preprocessor = TextPreprocessor(max_features=5000, ngram_range=(1, 2))
X_train_features = preprocessor.fit_transform(X_train)  # Fit + transform
X_test_features  = preprocessor.transform(X_test)       # Transform only
```

---

### Step 3: TF-IDF Vectorization
**File**: `src/preprocessing.py` → `TextPreprocessor.fit_transform()`

**TF-IDF** = Term Frequency × Inverse Document Frequency

- **TF**: How often a word appears in a message
- **IDF**: How rare the word is across all messages
- **n-grams**: Captures word pairs like "free prize", "call now"
- Produces a sparse matrix: `(n_samples, 5000)` features

Why TF-IDF beats raw word counts:
- Penalizes common words (like "the", "is")
- Rewards rare but informative words (like "prize", "urgent")

---

### Step 4: Model Training
**File**: `src/model.py` → `SpamClassifier`

**Logistic Regression** computes:
```
P(spam | features) = sigmoid(w·x + b)
```

- **Input**: 5,000-dimensional TF-IDF feature vector
- **Output**: Probability of spam (0 to 1)
- **Decision boundary**: 0.5 (tunable for precision/recall trade-off)

Why Logistic Regression for text:
- Works great with high-dimensional sparse features
- Fast to train and predict
- Coefficients directly interpretable (what words predict spam)

---

### Step 5: Evaluation
**File**: `src/evaluate.py`

| Metric | What it measures | Expected |
|--------|------------------|---------|
| **Accuracy** | Overall correct predictions | ~97.5% |
| **Precision** | Of predicted spam, how many actually are spam | ~99% |
| **Recall** | Of actual spam, how many did we catch | ~85% |
| **F1 Score** | Harmonic mean of precision and recall | ~92% |
| **ROC-AUC** | Ability to distinguish spam from ham | ~99% |

**Why high precision matters**: A false positive (labelling legitimate email as spam) is worse than missing some spam. Users lose important emails.

---

### Step 6: Model Persistence
**File**: `src/main.py`

Both the model and preprocessor are saved separately:

```
models/
├── spam_classifier.pkl   ← Trained LogisticRegression
└── preprocessor.pkl      ← Fitted TF-IDF vectorizer
```

Why save both? The vectorizer must use the **same vocabulary** learned during training. Loading just the model without the vectorizer would fail at inference time.

---

### Step 7: Flask API
**File**: `api/app.py`

```
Client Request
     │
     ▼  POST /predict
     │  {"message": "You won a free iPhone!"}
     │
     ▼
Load model + preprocessor (once at startup)
     │
     ▼
Clean text → TF-IDF features → Model.predict_proba()
     │
     ▼
     │  Response:
     └─► {
           "prediction": "SPAM",
           "confidence": 0.9823,
           "spam_probability": 0.9823,
           "is_spam": true
         }
```

---

## Data Flow Through Code

```
spam.csv
  └── load_and_prepare_data()
        ├── X_train (pd.Series) ─► TextPreprocessor.fit_transform()
        │                               └── X_train_tfidf (np.ndarray)
        │                                       └── SpamClassifier.fit()
        │                                               └── trained model
        │
        └── X_test (pd.Series) ─► TextPreprocessor.transform()
                                        └── X_test_tfidf (np.ndarray)
                                                └── SpamClassifier.predict()
                                                        └── metrics
```

---

## Running the Full Flow

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train (downloads data, trains, saves model)
python src/main.py

# Step 3: Start API
python api/app.py

# Step 4: Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won a free gift card. Call now!"}'
```

---

## Key Learnings

| Concept | Where Used |
|---------|-----------|
| TF-IDF | Text → numbers conversion |
| Train/test split with stratification | Preserve class ratio |
| Precision vs Recall trade-off | Email spam has high precision cost |
| Model + preprocessor serialization | Consistent inference |
| REST API design | Real-world deployment pattern |
