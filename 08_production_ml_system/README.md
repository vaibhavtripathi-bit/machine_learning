# Production ML System

End-to-end production machine learning system with training pipelines, model serving, monitoring, and CI/CD.

## Features

- **MLflow Tracking**: Experiment tracking and model registry
- **FastAPI Serving**: High-performance REST API
- **Docker Deployment**: Containerized application
- **Prometheus Metrics**: Monitoring and alerting
- **Drift Detection**: Automated data drift monitoring
- **CI/CD Pipeline**: GitHub Actions automation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python training/train.py

# Start API server
python serving/api/app.py

# Or use Docker
docker-compose -f serving/docker/docker-compose.yml up
```

## Project Structure

```
08_production_ml_system/
├── training/
│   ├── train.py              # Training pipeline with MLflow
│   └── pipelines/            # Additional training pipelines
├── serving/
│   ├── api/
│   │   └── app.py            # FastAPI server
│   ├── model/                # Trained models
│   └── docker/
│       ├── Dockerfile
│       ├── docker-compose.yml
│       └── prometheus.yml
├── monitoring/
│   ├── drift_detection.py    # Data drift detection
│   └── metrics/              # Custom metrics
├── ci_cd/
│   └── github_actions.yml    # CI/CD pipeline
├── data/                     # Training data
├── requirements.txt
└── README.md
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Training   │────►│   MLflow    │────►│   Model     │
│  Pipeline   │     │   Tracking  │     │   Registry  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────►│  FastAPI    │────►│   Model     │
│   Request   │     │   Server    │     │  Inference  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                           │                    │
                           ▼                    │
                    ┌─────────────┐             │
                    │ Prometheus  │◄────────────┘
                    │   Metrics   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Grafana   │
                    │  Dashboard  │
                    └─────────────┘
```

## Training Pipeline

### Train with MLflow

```bash
# Train and track experiments
python training/train.py

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Experiment Tracking

MLflow tracks:
- Parameters (hyperparameters)
- Metrics (accuracy, F1, etc.)
- Artifacts (model files)
- Model versions

## Model Serving

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Make predictions |
| `/metrics` | GET | Prometheus metrics |
| `/model/info` | GET | Model metadata |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

Response:
```json
{
  "predictions": [0],
  "probabilities": [[0.98, 0.01, 0.01]],
  "labels": ["setosa"],
  "model_name": "iris_classifier",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t ml-api -f serving/docker/Dockerfile .

# Run container
docker run -p 8000:8000 ml-api

# Or use docker-compose (includes Prometheus + Grafana)
docker-compose -f serving/docker/docker-compose.yml up
```

### Services

- **ml-api**: Model serving (port 8000)
- **prometheus**: Metrics collection (port 9090)
- **grafana**: Dashboards (port 3000, admin/admin)

## Monitoring

### Drift Detection

```python
from monitoring.drift_detection import DriftDetector

detector = DriftDetector(training_data, feature_names)
results = detector.get_summary(new_data)

if results['summary']['requires_retraining']:
    trigger_retraining()
```

### Prometheus Metrics

- `predictions_total`: Total predictions by status
- `prediction_latency_seconds`: Inference latency histogram

## CI/CD Pipeline

### GitHub Actions Workflow

1. **Test**: Run unit tests
2. **Train**: Retrain model (on demand)
3. **Build**: Build Docker image
4. **Deploy**: Deploy to production

Trigger retraining:
```bash
git commit -m "Update training data [retrain]"
```

## Key Concepts

1. **MLflow**: Experiment tracking, model registry
2. **FastAPI**: Async Python web framework
3. **Docker**: Containerization for deployment
4. **Prometheus**: Metrics and monitoring
5. **Data Drift**: Detecting distribution changes
6. **CI/CD**: Automated testing and deployment

## Extending the Project

- Add A/B testing for model comparison
- Implement shadow mode deployment
- Add automatic retraining triggers
- Set up Kubernetes deployment
- Add feature store integration

## License

MIT License
