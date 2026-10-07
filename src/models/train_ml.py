"""
Machine Learning Model Training & Comparative Benchmarking.
Trains and compares Random Forest, Gradient Boosting, Extra Trees, and Logistic Regression.
Evaluates accuracy, precision, recall, F1-score, ROC-AUC, FPR, FNR, and inference latency.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from src.features.feature_extractor import URLFeatureExtractor


def train_and_evaluate_models(
    dataset_path: str = "data/processed/features_dataset.csv",
    output_dir: str = "saved_models",
    random_state: int = 42
):
    """
    Loads extracted features, trains 4 classical ML classifiers, evaluates metrics,
    and saves the best model and performance reports.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading feature dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Feature columns (exclude target 'label' and raw string 'url')
    feature_cols = [c for c in df.columns if c not in ["label", "url"]]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"Features: {len(feature_cols)}, Samples: {X.shape[0]}")

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and feature names
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Dictionary of Candidate Models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=16, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=12, random_state=random_state
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120, max_depth=16, random_state=random_state, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        )
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    print("\n" + "=" * 65)
    print("           TRAINING & BENCHMARKING ML MODELS")
    print("=" * 65)

    roc_data = {}

    for name, model in models.items():
        print(f"\nTraining [{name}]...")
        t0 = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0

        # Inference Latency Benchmark
        t_infer_start = time.time()
        y_pred = model.predict(X_test_scaled)
        latency_ms = ((time.time() - t_infer_start) / len(X_test)) * 1000

        # Predict Probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = y_pred

        # Metric Calculations
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr_curve.tolist(), "tpr": tpr_curve.tolist()}

        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
            "False_Positive_Rate": round(fpr, 4),
            "False_Negative_Rate": round(fnr, 4),
            "Inference_Latency_ms": round(latency_ms, 4),
            "Train_Time_s": round(train_time, 4),
            "Confusion_Matrix": {
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
            }
        }

        print(f"  -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(f"  -> FPR: {fpr*100:.2f}% | FNR: {fnr*100:.2f}% | Latency: {latency_ms:.3f} ms/sample")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    print("\n" + "=" * 65)
    print(f"Top Performing Model: {best_model_name} (F1-Score: {best_f1:.4f})")
    print("=" * 65)

    # Save Best Model
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(best_model_obj, best_model_path)
    print(f"Saved best model to: {best_model_path}")

    # Feature Importance (Random Forest & Extra Trees)
    feature_importances = {}
    if hasattr(best_model_obj, "feature_importances_"):
        fi_vals = best_model_obj.feature_importances_
        sorted_indices = np.argsort(fi_vals)[::-1]
        for idx in sorted_indices:
            feature_importances[feature_cols[idx]] = float(round(fi_vals[idx], 4))
    elif hasattr(models["Random Forest"], "feature_importances_"):
        fi_vals = models["Random Forest"].feature_importances_
        sorted_indices = np.argsort(fi_vals)[::-1]
        for idx in sorted_indices:
            feature_importances[feature_cols[idx]] = float(round(fi_vals[idx], 4))

    with open(os.path.join(output_dir, "feature_importance.json"), "w") as f:
        json.dump(feature_importances, f, indent=2)

    # Save Metrics & ROC data
    summary = {
        "best_model_name": best_model_name,
        "test_sample_count": len(y_test),
        "train_sample_count": len(y_train),
        "models": results,
        "roc_curves": roc_data
    }

    with open(os.path.join(output_dir, "model_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Generate Evaluation Comparison Plot
    _generate_evaluation_plots(results, roc_data, output_dir)

    return results, best_model_name


def _generate_evaluation_plots(results: dict, roc_data: dict, output_dir: str):
    """Generates comparison bar charts and ROC curves for documentation/UI."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Bar Chart of Key Metrics
    names = list(results.keys())
    accs = [results[m]["Accuracy"] for m in names]
    f1s = [results[m]["F1_Score"] for m in names]
    aucs = [results[m]["ROC_AUC"] for m in names]

    x = np.arange(len(names))
    width = 0.25

    axes[0].bar(x - width, accs, width, label="Accuracy", color="#3498db")
    axes[0].bar(x, f1s, width, label="F1-Score", color="#2ecc71")
    axes[0].bar(x + width, aucs, width, label="ROC-AUC", color="#e74c3c")
    axes[0].set_ylabel("Score (0.0 - 1.0)")
    axes[0].set_title("Model Performance Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylim(0.7, 1.05)
    axes[0].legend(loc="lower right")

    # 2. ROC Curves
    for name, data in roc_data.items():
        axes[1].plot(data["fpr"], data["tpr"], label=f"{name} (AUC={results[name]['ROC_AUC']})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.6)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves Comparison")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_comparison_plots.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved evaluation plot to: {plot_path}")


if __name__ == "__main__":
    train_and_evaluate_models()

