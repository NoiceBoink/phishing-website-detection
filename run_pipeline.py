"""
Master End-to-End Execution Pipeline for Phishing Website Detection System.
Executes:
1. Benchmark Dataset Generation & Feature Extraction (32 Features)
2. Machine Learning Model Training & Comparative Benchmarking (Random Forest, Gradient Boosting, Extra Trees, Logistic Regression)
3. Deep Learning Training (Character-Level CNN-BiLSTM via PyTorch)
4. Model Export & Explainability Engine Setup
5. End-to-End Validation on Test Phishing & Benign URLs
"""

import os
import sys
import json
import time

from src.data.dataset_builder import prepare_dataset
from src.models.train_ml import train_and_evaluate_models
from src.models.train_dl import train_deep_learning_model
from src.models.predictor import PhishingPredictor


def run_full_pipeline(num_samples: int = 4000, dl_epochs: int = 5):
    start_time = time.time()
    print("=" * 70)
    print("     PHISHING WEBSITE DETECTION SYSTEM - AI/ML PIPELINE")
    print("=" * 70)

    # Step 1: Dataset Generation & Feature Extraction
    print("\n[STEP 1/4] Preparing Dataset & Extracting 32 Handcrafted Features...")
    raw_df, feat_df = prepare_dataset(
        output_dir="data",
        num_samples=num_samples,
        force_rebuild=False
    )
    print(f"Dataset Ready: {len(raw_df)} total URLs ({len(feat_df.columns)} feature columns).")

    # Step 2: Classical & Ensemble ML Training & Benchmarking
    print("\n[STEP 2/4] Training & Benchmarking Machine Learning Models...")
    ml_results, best_model_name = train_and_evaluate_models(
        dataset_path="data/processed/features_dataset.csv",
        output_dir="saved_models"
    )

    # Step 3: Deep Learning (Char-CNN-LSTM)
    print("\n[STEP 3/4] Training Character-Level CNN-BiLSTM Neural Network (PyTorch)...")
    try:
        train_deep_learning_model(
            raw_data_path="data/raw/urls.csv",
            output_dir="saved_models",
            epochs=dl_epochs,
            batch_size=64
        )
    except Exception as e:
        print(f"Warning: Deep learning training step encountered: {e}. Classical models remain active.")

    # Step 4: System Verification & Test Inferences
    print("\n[STEP 4/4] Verifying End-to-End Predictor & Explainability Engine...")
    predictor = PhishingPredictor(models_dir="saved_models")

    test_urls = [
        "https://www.google.com/search?q=machine+learning+tutorial",
        "https://github.com/torvalds/linux",
        "http://paypal.com.account-update-security.xyz/login.php",
        "http://192.168.1.105:8080/secure/chase/login.php",
        "http://netflix-billing-renew.icu/account/verify"
    ]

    print("\nRunning Test Inferences:")
    print("-" * 70)
    for test_url in test_urls:
        res = predictor.predict(test_url)
        print(f"URL:     {res['url']}")
        print(f"Verdict: {res['verdict']} (Risk: {res['risk_score']}%) | Latency: {res['latency_ms']} ms")
        if res['explanation']['natural_language_reasons']:
            print(f"Key Risk Factors: {', '.join(res['explanation']['natural_language_reasons'][:2])}")
        print("-" * 70)

    # Summary
    metrics_path = "saved_models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            summary = json.load(f)
        print("\n" + "=" * 70)
        print("                 BENCHMARK SUMMARY RESULTS")
        print("=" * 70)
        print(f"{'Model Name':<32} | {'Accuracy':<9} | {'F1-Score':<9} | {'ROC-AUC':<9}")
        print("-" * 70)
        for m_name, m_metrics in summary.get("models", {}).items():
            print(f"{m_name:<32} | {m_metrics.get('Accuracy', 0):<9.4f} | {m_metrics.get('F1_Score', 0):<9.4f} | {m_metrics.get('ROC_AUC', 0):<9.4f}")
        print("=" * 70)

    total_time = round(time.time() - start_time, 2)
    print(f"\nPipeline successfully completed in {total_time} seconds!")
    print("Run `streamlit run app.py` to start the interactive web application.")
    print("Run `uvicorn api:app --reload` to start the FastAPI REST service.")


if __name__ == "__main__":
    run_full_pipeline()

