# 🛡️ PhishShield AI - Phishing Website Detection System
### Applied AI/ML Academic Project: Dual-Paradigm Detection & Explainable AI (XAI)

PhishShield AI is an intelligent cybersecurity system designed to identify and explain malicious phishing URLs using a dual-paradigm machine learning framework. It combines **32 handcrafted lexical, structural, and security heuristic features** with a **Character-Level Deep Learning Convolutional & Recurrent Neural Network (Char-CNN-BiLSTM)**.

---

## 🌟 Key Features

1. **Dual-Paradigm Architecture**:
   - **Handcrafted ML Models**: Random Forest, HistGradientBoosting, Extra Trees, and Logistic Regression.
   - **Sequence Deep Learning**: 1D-CNN + Bidirectional LSTM in PyTorch capturing raw character n-grams and long-range syntactic patterns without manual feature engineering.
2. **32 Comprehensive Extracted Features**:
   - **Lexical/Length**: URL length, domain length, path length, query length, TLD length.
   - **Syntactic Delimiters**: Counts of `.`, `-`, `_`, `/`, `//`, `@`, `?`, `=`, `%`, `&`, digits, and letter/symbol ratios.
   - **Statistical & Information Theory**: Shannon entropy of URL string & domain.
   - **Security Heuristics**: IPv4/IPv6 hostname detection, nested subdomain depth, HTTPS token spoofing, high-risk TLDs, URL shorteners, credential-harvesting keyword counts, Punycode homoglyphs, and non-standard ports.
3. **Explainable AI (XAI)**:
   - Sample-level attribution explaining *why* an individual URL was classified as phishing.
   - Outputs top threat contributors and mitigating factors in plain English.
4. **Interactive Interfaces**:
   - **Streamlit Web Application** (`app.py`): Live URL testing, risk gauge, model benchmark comparison, and batch CSV scanner.
   - **FastAPI REST Service** (`api.py`): Production-ready API for browser extension or microservice integration.

---

## 📂 Project Structure

```text
aiml project/
├── data/
│   ├── raw/                  # Raw benchmark URLs (urls.csv)
│   └── processed/            # Extracted 32-feature matrix (features_dataset.csv)
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_extractor.py  # 32-feature extraction engine
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_builder.py    # Dataset generator & preprocessor
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_ml.py           # Scikit-Learn training & benchmarking
│   │   ├── train_dl.py           # PyTorch Char-CNN-BiLSTM deep learning
│   │   └── predictor.py          # Unified inference & prediction engine
│   └── explainability/
│       ├── __init__.py
│       └── explainer.py          # Global & Local XAI attribution
├── saved_models/
│   ├── best_model.joblib         # Top-performing ML model artifact
│   ├── scaler.joblib             # Standard scaler weights
│   ├── char_cnn_lstm.pt          # PyTorch weights
│   ├── char_vocab.json           # Character tokenizer vocabulary
│   ├── feature_names.json        # Ordered list of feature names
│   ├── feature_importance.json   # Global feature ranking
│   ├── model_metrics.json        # Quantitative benchmark results
│   └── model_comparison_plots.png# Metric bar charts & ROC curves
├── app.py                        # Streamlit Interactive Web Application
├── api.py                        # FastAPI REST API Server
├── run_pipeline.py               # Master end-to-end pipeline runner
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation
└── AIML_PROJECT_REPORT.md        # Academic report & viva cheat sheet
```

---

## 🚀 Quickstart Guide

### 1. Run the End-to-End Pipeline
Executes dataset generation, feature extraction, ML training, Deep Learning training, benchmarking, and validation in a single command:
```bash
python run_pipeline.py
```

### 2. Launch the Interactive Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Start the FastAPI REST Server
```bash
uvicorn api:app --reload
```
Access interactive Swagger API docs at `http://localhost:8000/docs`.

---

## 📊 Benchmark Metrics Tracked
- **Accuracy**
- **Precision** (Crucial to prevent false alarms on legitimate banking sites)
- **Recall** (Crucial to catch sophisticated phishing attacks)
- **F1-Score** (Balanced metric for class representation)
- **ROC-AUC** (Area under receiver operating characteristic curve)
- **False Positive Rate (FPR)**
- **False Negative Rate (FNR)**
- **Inference Latency (ms/sample)**

