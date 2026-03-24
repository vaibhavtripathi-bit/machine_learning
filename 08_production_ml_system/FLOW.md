# End-to-End Flow: Production ML System

## Overview

This document explains the complete pipeline of a production-grade ML system — from experiment tracking through model serving, monitoring, and automated CI/CD.

---

## Flow Diagram

```
New Data / Code Change
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│                   CI/CD PIPELINE                          │
│  (GitHub Actions: .github/workflows/ml-pipeline.yml)     │
│                                                           │
│   1. Tests pass? → 2. Retrain trigger? → 3. Build image  │
└──────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────┐
│  Training Pipeline   │  ← training/train.py
│  (MLflow tracked)    │    Logs params, metrics, artifacts
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  MLflow Experiment   │  ← Compare runs
│  Tracking            │    Pick best model
│  (sqlite:///mlflow)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Model Registry      │  ← Register best model
│  (MLflow)            │    Version tracking
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Containerize        │  ← Docker build
│  (Dockerfile)        │    serving/docker/
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│                    PRODUCTION SERVING                     │
│                                                           │
│   FastAPI Server                                          │
│     ├── POST /predict    → model inference               │
│     ├── GET  /health     → liveness check                │
│     ├── GET  /metrics    → Prometheus scrape target      │
│     └── GET  /model/info → metadata                      │
└──────────┬───────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│                    MONITORING                             │
│                                                           │
│   Prometheus ← scrapes /metrics                          │
│      └── Grafana ← visualizes dashboards                 │
│                                                           │
│   Drift Detector ← checks production data distribution   │
│      └── Alert if drift detected → trigger retrain       │
└──────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Breakdown

### Step 1: Training with MLflow
**File**: `training/train.py` → `run_experiment()`

MLflow tracks every experiment automatically:

```python
with mlflow.start_run(run_name="run_1"):
    mlflow.log_params({"n_estimators": 100, "max_depth": 5})  # Hyperparams
    
    model = train_model(X_train, y_train, params)
    metrics = evaluate_model(model, X_test, y_test)
    
    mlflow.log_metrics({"accuracy": 0.97, "f1": 0.96})        # Results
    mlflow.sklearn.log_model(model, "model")                   # Artifact
```

**MLflow stores**:
```
mlruns/
└── experiment_id/
    └── run_id/
        ├── params/       ← n_estimators=100, max_depth=5
        ├── metrics/      ← accuracy, f1, precision over time
        └── artifacts/    ← model.pkl, plots, confusion matrix
```

---

### Step 2: Model Comparison

3 experiments run with different hyperparameters:

```
Run 1: n_estimators=50,  max_depth=3  → accuracy=0.93
Run 2: n_estimators=100, max_depth=5  → accuracy=0.96  ← best
Run 3: n_estimators=200, max_depth=10 → accuracy=0.95  (overfitting)
```

MLflow UI shows all runs side by side:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open: http://localhost:5000
```

---

### Step 3: Model Registration

The best run's model is registered in the **MLflow Model Registry**:

```
Model Registry: "iris_classifier"
    ├── Version 1  (Run: abc123)  → Staging
    ├── Version 2  (Run: def456)  → Production ← current
    └── Version 3  (Run: ghi789)  → Archived
```

This provides:
- Version history
- Stage transitions (Staging → Production)
- Annotations and descriptions
- Easy rollback

---

### Step 4: FastAPI Serving
**File**: `serving/api/app.py`

The server loads the trained model at startup:

```
Server starts
    │
    ▼  load_model() → joblib.load("model/model.joblib")
    │                 json.load("model/metadata.json")
    │
    ▼  Model ready in memory
```

**Request Flow**:
```
Client: POST /predict
        {"features": [[5.1, 3.5, 1.4, 0.2]]}
          │
          ▼  Validate with Pydantic schema
          │
          ▼  X = np.array(features)
          │
          ▼  predictions = model.predict(X)
          │  probabilities = model.predict_proba(X)
          │
          ▼  Map predictions to label names
          │  0 → "setosa", 1 → "versicolor", 2 → "virginica"
          │
          ▼  Response:
          {
            "predictions": [0],
            "probabilities": [[0.98, 0.01, 0.01]],
            "labels": ["setosa"],
            "model_name": "iris_classifier",
            "timestamp": "2024-01-15T10:30:00"
          }
```

**FastAPI advantages**:
- Auto-generates API documentation at `/docs`
- Request/response validation with Pydantic
- Async support for high concurrency
- Native OpenAPI schema

---

### Step 5: Containerization
**File**: `serving/docker/Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serving/api/ ./api/
COPY serving/model/ ./model/
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker benefits**:
- Reproducible environment (no "works on my machine")
- Easy horizontal scaling
- Isolated from host system

**docker-compose.yml** starts the full stack:
```yaml
services:
  ml-api:      # Model server (port 8000)
  prometheus:  # Metrics collection (port 9090)
  grafana:     # Dashboards (port 3000)
```

---

### Step 6: Prometheus Metrics
**File**: `serving/api/app.py`

Custom metrics are exposed at `/metrics`:

```python
PREDICTIONS_TOTAL = Counter(
    'predictions_total',
    'Total predictions',
    ['model_name', 'status']         # Labels for filtering
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Inference latency',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.5]  # Bucket boundaries
)
```

Prometheus scrapes every 15 seconds:
```
GET http://ml-api:8000/metrics
→ # HELP predictions_total Total predictions
→ predictions_total{model_name="iris_classifier",status="success"} 1234
→ prediction_latency_seconds_p95 0.023
```

**Grafana dashboards** visualize:
- Requests per second
- P50/P95/P99 latency
- Error rate
- Model accuracy over time

---

### Step 7: Data Drift Detection
**File**: `monitoring/drift_detection.py` → `DriftDetector`

Production data can drift from training data over time:

```
Training data (2023):   avg_feature_1 = 5.1 ± 0.8
Production data (2024): avg_feature_1 = 6.3 ± 1.2  ← Drifted!
```

**KS Test** (Kolmogorov-Smirnov):
```
H₀: Both samples from the same distribution
p-value < 0.05 → Reject H₀ → Drift detected!

For each feature:
  ks_statistic, p_value = ks_2samp(reference[:, i], new[:, i])
```

**PSI** (Population Stability Index):
```
PSI = Σ (new_% - ref_%) × ln(new_% / ref_%)

PSI < 0.1  → No significant change
PSI 0.1-0.2 → Moderate change, monitor
PSI > 0.2  → Significant change, retrain!
```

---

### Step 8: CI/CD with GitHub Actions
**File**: `ci_cd/github_actions.yml`

```
Push to main branch
        │
        ▼  test job
        │  - Install dependencies
        │  - pytest tests/
        │  - Coverage report
        │
        ▼  train job (only if [retrain] in commit message)
        │  - python training/train.py
        │  - Upload model artifact
        │
        ▼  build job
        │  - docker build + push to ghcr.io
        │
        ▼  deploy job (main branch only)
           - Deploy to production environment
```

**Trigger retraining**:
```bash
git commit -m "Update training data [retrain]"
git push
# → GitHub Actions runs full pipeline including training
```

---

## End-to-End Request Lifecycle

```
1. Client sends POST /predict with features
       │
2. FastAPI validates request with Pydantic
       │
3. Model inference: ~20ms
       │
4. Prometheus metrics updated:
       predictions_total{status="success"} += 1
       prediction_latency_seconds observed
       │
5. Response returned to client
       │
6. Background: Drift detector checks
       if production dist ≠ training dist:
           trigger_retraining_pipeline()
```

---

## Running the Full Flow

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Train model
cd training && python train.py
# → MLflow tracks 3 experiments
# → Best model saved to serving/model/

# Step 3: Start API
cd serving/api && python app.py
# → http://localhost:8000/docs   (Swagger UI)

# Step 4: Test endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'

# Step 5: Full stack with monitoring
docker-compose -f serving/docker/docker-compose.yml up
# → API:        http://localhost:8000
# → MLflow UI:  http://localhost:5000
# → Prometheus: http://localhost:9090
# → Grafana:    http://localhost:3000 (admin/admin)

# Step 6: Test drift detection
python monitoring/drift_detection.py
```

---

## Key Learnings

| Concept | Where Used |
|---------|-----------|
| MLflow experiment tracking | Log params, metrics, artifacts per run |
| Model registry | Versioned model lifecycle management |
| Pydantic validation | Request/response schema enforcement |
| Docker containerization | Reproducible deployment |
| Prometheus metrics | Latency, throughput, error rate |
| KS test / PSI | Statistical drift detection |
| GitHub Actions | Automated test → train → deploy pipeline |
| Health checks | Kubernetes/load balancer readiness probes |
