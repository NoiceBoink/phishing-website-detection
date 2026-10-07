"""
FastAPI REST API for Phishing Website Detection System.
Exposes endpoints for single URL prediction, batch evaluation,
feature extraction, and Explainable AI (XAI) diagnostics.
"""

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from src.models.predictor import PhishingPredictor
from src.features.feature_extractor import URLFeatureExtractor

app = FastAPI(
    title="PhishShield AI API",
    description="High-performance AI/ML REST API for Phishing Website Threat Detection and Explainability",
    version="1.0.0"
)

_predictor = None
_extractor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = PhishingPredictor(models_dir="saved_models")
        except Exception as e:
            print(f"Error loading predictor: {e}")
    return _predictor


def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = URLFeatureExtractor()
    return _extractor


class URLRequest(BaseModel):
    url: str

class BatchURLRequest(BaseModel):
    urls: List[str]

class PredictionResponse(BaseModel):
    url: str
    verdict: str
    severity: str
    is_phishing: bool
    risk_score: float
    ml_probability: float
    dl_probability: Optional[float] = None
    latency_ms: float
    explanation: Dict[str, Any]

class BatchResponse(BaseModel):
    total_scanned: int
    results: List[Dict[str, Any]]


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PhishShield AI - Phishing Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Analyze single URL with ML & XAI",
            "/predict-batch": "POST - Analyze multiple URLs",
            "/extract-features": "POST - Return 32 extracted feature values",
            "/health": "GET - System health and model status"
        }
    }


@app.get("/health")
def health():
    p = get_predictor()
    return {
        "status": "healthy",
        "models_loaded": p is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_url(req: URLRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    p = get_predictor()
    if p is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run pipeline first.")
    
    result = p.predict(req.url.strip())
    return {
        "url": result["url"],
        "verdict": result["verdict"],
        "severity": result["severity"],
        "is_phishing": result["is_phishing"],
        "risk_score": result["risk_score"],
        "ml_probability": result["ml_probability"],
        "dl_probability": result["dl_probability"],
        "latency_ms": result["latency_ms"],
        "explanation": result["explanation"]
    }


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(req: BatchURLRequest):
    if not req.urls:
        raise HTTPException(status_code=400, detail="URL list cannot be empty")
    p = get_predictor()
    if p is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run pipeline first.")

    results = []
    for u in req.urls:
        if u.strip():
            res = p.predict(u.strip())
            results.append({
                "url": res["url"],
                "verdict": res["verdict"],
                "risk_score": res["risk_score"],
                "ml_probability": res["ml_probability"],
                "latency_ms": res["latency_ms"]
            })

    return {
        "total_scanned": len(results),
        "results": results
    }


@app.post("/extract-features")
def extract_features(req: URLRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    ext = get_extractor()
    features = ext.extract_features(req.url.strip())
    return {
        "url": req.url,
        "feature_count": len(features),
        "features": features
    }

