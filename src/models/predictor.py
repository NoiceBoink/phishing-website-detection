"""
Unified Predictor and Inference Engine for Phishing Detection.
Wraps feature extraction, ML/DL models, and Explainable AI into a clean interface.
"""

import os
import time
import json
import joblib
import numpy as np
from typing import Dict, Any, List, Optional

from src.features.feature_extractor import URLFeatureExtractor
from src.explainability.explainer import PhishingExplainer


class PhishingPredictor:
    """
    End-to-end predictor for evaluating URLs.
    Handles feature extraction, model prediction, risk scoring, and explainability.
    """

    def __init__(self, models_dir: str = "saved_models"):
        self.models_dir = models_dir
        self.extractor = URLFeatureExtractor()

        # Load Scaler
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training pipeline first.")
        self.scaler = joblib.load(scaler_path)

        # Load Feature Names
        feat_path = os.path.join(models_dir, "feature_names.json")
        with open(feat_path, "r") as f:
            self.feature_names = json.load(f)

        # Load Best ML Model
        model_path = os.path.join(models_dir, "best_model.joblib")
        self.ml_model = joblib.load(model_path)

        # Initialize Explainer
        self.explainer = PhishingExplainer(
            model=self.ml_model,
            scaler=self.scaler,
            feature_names=self.feature_names
        )

        # Attempt to load PyTorch DL model if present
        self.dl_model = None
        dl_path = os.path.join(models_dir, "char_cnn_lstm.pt")
        vocab_path = os.path.join(models_dir, "char_vocab.json")
        if os.path.exists(dl_path) and os.path.exists(vocab_path):
            try:
                import torch
                from src.models.train_dl import CharCNNLSTM, encode_url
                self.torch = torch
                self.encode_url = encode_url
                self.dl_model = CharCNNLSTM()
                self.dl_model.load_state_dict(torch.load(dl_path, map_location=torch.device("cpu")))
                self.dl_model.eval()
            except Exception as e:
                print(f"Notice: PyTorch DL model could not be initialized ({e}). ML model active.")

    def predict(self, url: str) -> Dict[str, Any]:
        """
        Runs full detection pipeline on a single URL string.
        """
        t0 = time.time()
        raw_url = str(url).strip()

        # 1. Feature Extraction
        features = self.extractor.extract_features(raw_url)

        # 2. Vectorize & Scale
        feat_vector = np.array([features[c] for c in self.feature_names], dtype=float).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat_vector)

        # 3. Predict ML Probability
        if hasattr(self.ml_model, "predict_proba"):
            ml_prob = float(self.ml_model.predict_proba(feat_scaled)[0, 1])
        else:
            ml_prob = float(self.ml_model.predict(feat_scaled)[0])

        # 4. Predict DL Probability (if loaded)
        dl_prob = None
        if self.dl_model is not None:
            try:
                encoded = self.encode_url(raw_url)
                tensor_in = self.torch.tensor(encoded, dtype=self.torch.long).unsqueeze(0)
                with self.torch.no_grad():
                    logits = self.dl_model(tensor_in)
                    dl_prob = float(self.torch.sigmoid(logits).item())
            except Exception:
                dl_prob = None

        # 5. Combined / Primary Probability
        # If DL is available, ensemble blend (70% ML tabular + 30% DL char sequence)
        if dl_prob is not None:
            final_prob = float(0.70 * ml_prob + 0.30 * dl_prob)
        else:
            final_prob = ml_prob

        # Thresholds
        risk_score = round(final_prob * 100, 2)
        if final_prob >= 0.65:
            verdict = "PHISHING"
            severity = "CRITICAL"
        elif final_prob >= 0.35:
            verdict = "SUSPICIOUS"
            severity = "WARNING"
        else:
            verdict = "SAFE"
            severity = "SAFE"

        # 6. Local Explainability
        explanation = self.explainer.explain_prediction(
            feature_dict=features,
            base_probability=ml_prob,
            top_k=5
        )

        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "url": raw_url,
            "verdict": verdict,
            "severity": severity,
            "is_phishing": bool(final_prob >= 0.50),
            "risk_score": risk_score,
            "ml_probability": round(ml_prob, 4),
            "dl_probability": round(dl_prob, 4) if dl_prob is not None else None,
            "features": features,
            "explanation": explanation,
            "latency_ms": latency_ms
        }

    def predict_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Processes a list of URLs."""
        return [self.predict(u) for u in urls]

