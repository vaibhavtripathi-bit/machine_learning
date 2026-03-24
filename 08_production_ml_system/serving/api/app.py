"""
FastAPI server for model serving.
"""

import os
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response


app = FastAPI(
    title="ML Model API",
    description="Production ML model serving with FastAPI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTIONS_TOTAL = Counter(
    'predictions_total', 
    'Total number of predictions',
    ['model_name', 'status']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name']
)

model = None
metadata = None


class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[List[float]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: List[int]
    probabilities: List[List[float]]
    labels: List[str]
    model_name: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str]


def load_model():
    """Load the trained model."""
    global model, metadata
    
    model_dir = Path(__file__).parent.parent / "model"
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Train a model first using: python training/train.py")
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Metadata loaded: {metadata}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_name=metadata.get("model_name") if metadata else None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions."""
    if model is None:
        PREDICTIONS_TOTAL.labels(model_name="none", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    try:
        X = np.array(request.features)
        
        predictions = model.predict(X).tolist()
        probabilities = model.predict_proba(X).tolist()
        
        target_names = metadata.get("target_names", []) if metadata else []
        labels = [target_names[p] if p < len(target_names) else str(p) for p in predictions]
        
        latency = time.time() - start_time
        PREDICTION_LATENCY.labels(model_name=metadata.get("model_name", "unknown")).observe(latency)
        PREDICTIONS_TOTAL.labels(model_name=metadata.get("model_name", "unknown"), status="success").inc()
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            labels=labels,
            model_name=metadata.get("model_name", "unknown") if metadata else "unknown",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        PREDICTIONS_TOTAL.labels(model_name=metadata.get("model_name", "unknown"), status="error").inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if metadata is None:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    return metadata


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
