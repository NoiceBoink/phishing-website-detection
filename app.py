"""
Streamlit Web Dashboard for Phishing Website Detection System.
Provides Real-Time URL Scanning, Explainable AI (XAI) Attribution,
Comparative Model Benchmarking, and Batch URL Analysis.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

# Configure page
st.set_page_config(
    page_title="PhishShield AI - Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-safe {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #713F12;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-danger {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Cached loader for predictor model."""
    from src.models.predictor import PhishingPredictor
    return PhishingPredictor(models_dir="saved_models")


@st.cache_data
def load_metrics():
    """Load benchmark metrics JSON."""
    metrics_path = "saved_models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_feature_importances():
    """Load global feature importance rankings."""
    path = "saved_models/feature_importance.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.markdown("### **PhishShield AI**")
    st.caption("AI/ML Phishing Detection System")
    st.markdown("---")
    st.markdown("**Core Capabilities:**")
    st.markdown("- 🔍 32 Handcrafted Security Features")
    st.markdown("- 🌳 Ensemble Machine Learning")
    st.markdown("- 🧠 1D-CNN + BiLSTM Deep Learning")
    st.markdown("- 💡 Explainable AI (XAI) Attribution")
    st.markdown("---")
    st.info("System Ready | Models Loaded")


# Header
st.markdown('<div class="main-title">🛡️ Phishing Website Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Modal Threat Identification powered by Classical Ensemble Learning, Deep Neural Sequences, and Explainable AI</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Real-Time URL Scanner",
    "📊 Model Benchmarking & Evaluation",
    "📁 Batch URL Analysis",
    "📐 Architecture & Methodology"
])


# ----------------------------------------------------
# TAB 1: Real-Time URL Scanner
# ----------------------------------------------------
with tab1:
    st.subheader("Live URL Threat Analysis")

    # Sample URL shortcuts
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    preset_url = ""
    if sample_col1.button("🟢 Sample Benign: Google Search"):
        preset_url = "https://www.google.com/search?q=machine+learning+tutorial"
    if sample_col2.button("🟢 Sample Benign: GitHub"):
        preset_url = "https://github.com/torvalds/linux/commit/master"
    if sample_col3.button("🔴 Sample Phish: PayPal Spoof"):
        preset_url = "http://paypal.com.account-verification-login.xyz/webscr"
    if sample_col4.button("🔴 Sample Phish: Direct IP Bank"):
        preset_url = "http://192.168.1.105:8080/secure/chase/login.php"

    url_input = st.text_input(
        "Enter Target URL to inspect:",
        value=preset_url if preset_url else "http://paypal.com.account-verification-login.xyz/webscr",
        placeholder="https://example.com/path"
    )

    if st.button("🚀 Analyze URL", type="primary", use_container_width=True):
        if not url_input.strip():
            st.warning("Please provide a valid URL string to analyze.")
        else:
            try:
                predictor = load_predictor()
                with st.spinner("Extracting 32 security features & evaluating AI models..."):
                    result = predictor.predict(url_input)

                # High-level Verdict Card
                risk_score = result["risk_score"]
                verdict = result["verdict"]

                st.markdown("---")
                v_col1, v_col2, v_col3, v_col4 = st.columns(4)

                with v_col1:
                    if verdict == "PHISHING":
                        st.markdown('<div class="badge-danger">🚨 PHISHING DETECTED</div>', unsafe_allow_html=True)
                    elif verdict == "SUSPICIOUS":
                        st.markdown('<div class="badge-warning">⚠️ SUSPICIOUS SITE</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="badge-safe">✅ LEGITIMATE (SAFE)</div>', unsafe_allow_html=True)

                with v_col2:
                    st.metric(label="Overall Threat Score", value=f"{risk_score}%")
                with v_col3:
                    st.metric(label="ML Model Probability", value=f"{result['ml_probability'] * 100:.1f}%")
                with v_col4:
                    st.metric(label="Inference Latency", value=f"{result['latency_ms']} ms")

                # Progress gauge
                st.progress(float(risk_score) / 100.0)

                # Explainable AI Section
                st.markdown("### 💡 Explainable AI (XAI) Diagnostics")
                st.caption("Detailed instance-level attribution explaining WHY this URL received this classification:")

                exp_data = result["explanation"]
                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    st.markdown("##### 🚨 Top Threat Contributors (Pushed Risk Up)")
                    pos_factors = exp_data.get("top_positive_contributors", [])
                    if pos_factors:
                        for item in pos_factors:
                            st.error(f"**{item['feature']}** = `{item['value']}` (+{item['impact']*100:.1f}% Risk)\n\n_{item['description']}_")
                    else:
                        st.success("No significant malicious anomalies detected.")

                with col_exp2:
                    st.markdown("##### 🛡️ Mitigating / Safe Factors (Reduced Risk)")
                    neg_factors = exp_data.get("top_mitigating_factors", [])
                    if neg_factors:
                        for item in neg_factors:
                            st.success(f"**{item['feature']}** = `{item['value']}` ({item['impact']*100:.1f}% Risk)\n\n_{item['description']}_")
                    else:
                        st.info("No mitigating safety factors found.")

                if exp_data.get("natural_language_reasons"):
                    st.markdown("##### 📝 Plain-English Summary")
                    for reason in exp_data["natural_language_reasons"]:
                        st.markdown(f"- {reason}")

                # Extracted Features Table
                with st.expander("🔬 View All 32 Extracted Features"):
                    feat_df = pd.DataFrame(list(result["features"].items()), columns=["Feature Name", "Extracted Value"])
                    st.dataframe(feat_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.info("Ensure the training pipeline has executed first by running `python run_pipeline.py`.")


# ----------------------------------------------------
# TAB 2: Model Benchmarking & Evaluation
# ----------------------------------------------------
with tab2:
    st.subheader("Model Performance & Comparative Evaluation")
    st.caption("Quantitative comparison across Classical ML, Ensemble Learners, and Deep Learning.")

    metrics = load_metrics()
    if metrics:
        models_data = metrics.get("models", {})
        
        # Build comparison table
        table_rows = []
        for m_name, m_stats in models_data.items():
            table_rows.append({
                "Model Architecture": m_name,
                "Accuracy": f"{m_stats.get('Accuracy', 0):.4f}",
                "Precision": f"{m_stats.get('Precision', 0):.4f}",
                "Recall": f"{m_stats.get('Recall', 0):.4f}",
                "F1-Score": f"{m_stats.get('F1_Score', 0):.4f}",
                "ROC-AUC": f"{m_stats.get('ROC_AUC', 0):.4f}",
                "Latency (ms)": f"{m_stats.get('Inference_Latency_ms', 0):.3f}",
                "Train Time (s)": f"{m_stats.get('Train_Time_s', 0):.2f}"
            })

        df_bench = pd.DataFrame(table_rows)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

        st.markdown(f"**Best Performing Model**: `{metrics.get('best_model_name', 'N/A')}`")

        # Visualization
        chart_df = pd.DataFrame([
            {
                "Model": m_name,
                "Accuracy": m_stats.get("Accuracy", 0),
                "F1-Score": m_stats.get("F1_Score", 0),
                "ROC-AUC": m_stats.get("ROC_AUC", 0)
            }
            for m_name, m_stats in models_data.items()
        ]).set_index("Model")

        st.bar_chart(chart_df)

        # Global Feature Importance Chart
        feat_importances = load_feature_importances()
        if feat_importances:
            st.markdown("### 🏆 Global Top 10 Most Predictive Features")
            top10_df = pd.DataFrame(
                list(feat_importances.items())[:10],
                columns=["Feature", "Gini Importance"]
            ).set_index("Feature")
            st.bar_chart(top10_df)

    else:
        st.warning("No benchmark data found. Run `python run_pipeline.py` to train models and generate metrics.")


# ----------------------------------------------------
# TAB 3: Batch URL Analysis
# ----------------------------------------------------
with tab3:
    st.subheader("Batch URL Scanner")
    st.caption("Paste multiple URLs (one per line) to evaluate threat scores simultaneously.")

    default_batch = (
        "https://www.google.com\n"
        "http://paypal.com.verification-center.xyz/login\n"
        "https://en.wikipedia.org/wiki/Phishing\n"
        "http://192.168.1.1:8080/auth\n"
        "http://netflix-billing-renew.icu/account"
    )
    batch_input = st.text_area("Input URLs:", value=default_batch, height=140)

    if st.button("⚡ Scan All URLs", type="primary"):
        urls = [u.strip() for u in batch_input.strip().split("\n") if u.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            predictor = load_predictor()
            results = []
            with st.spinner(f"Evaluating {len(urls)} URLs..."):
                for u in urls:
                    res = predictor.predict(u)
                    results.append({
                        "URL": res["url"],
                        "Verdict": res["verdict"],
                        "Risk Score (%)": res["risk_score"],
                        "ML Prob": round(res["ml_probability"], 3),
                        "Latency (ms)": res["latency_ms"]
                    })
            
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            phish_count = sum(1 for r in results if r["Verdict"] == "PHISHING")
            st.info(f"Scan Complete: {phish_count} / {len(urls)} URLs flagged as Phishing.")


# ----------------------------------------------------
# TAB 4: Architecture & Methodology
# ----------------------------------------------------
with tab4:
    st.subheader("System Architecture & AI/ML Methodology")
    
    st.markdown("""
    #### 🏗️ Dual-Paradigm Pipeline Architecture
    ```
    Raw URL String
         ├── [Path A: Handcrafted Feature Engineering]
         │    ├── Lexical Extraction (length, ratios, entropy)
         │    ├── Structural/Syntactic Parsing (delimiters, IP, subdomains)
         │    ├── Security Heuristics (suspicious TLDs, keywords, shorteners)
         │    └── Standard Scaling -> Random Forest / Gradient Boosting -> P(Phish)_ML
         │
         ├── [Path B: Character-Level Sequence Learning]
         │    ├── Character Vocabulary Tokenizer (128 ASCII dimensions)
         │    ├── 1D Convolutional Neural Network (Local n-gram extraction)
         │    └── Bidirectional LSTM (Sequential context) -> P(Phish)_DL
         │
         └── [Decision Fusion & Explainable AI (XAI)]
              ├── Ensemble Probability Calibration
              ├── Local Perturbation Feature Attribution
              └── Threat Verdict: SAFE | SUSPICIOUS | PHISHING
    ```

    #### 🔬 32 Extracted Features Breakdown
    - **Lexical/Length (5)**: URL length, domain length, path length, query length, TLD length.
    - **Syntactic/Character Counts (15)**: Dots, hyphens, underscores, slashes, double-slashes, `@`, `?`, `=`, `%`, `&`, digits, digit ratio, letter ratio, special character ratio, uppercase ratio.
    - **Statistical & Information Theory (2)**: Shannon entropy of URL string, Shannon entropy of domain.
    - **Security & Heuristics (10)**: IP address detection, subdomain depth count, HTTPS protocol, deceptive HTTPS in domain, abusive TLD list, URL shorteners, phishing keywords count, Punycode homoglyph check, non-standard port check, client/server token.

    #### 📐 Academic Evaluation Metrics
    - **Precision**: $\\frac{TP}{TP + FP}$ (Minimizes false alarms on legitimate banking portals)
    - **Recall**: $\\frac{TP}{TP + FN}$ (Maximizes catching malicious sites before user credentials leak)
    - **F1-Score**: Harmonic mean of Precision and Recall
    - **False Positive Rate (FPR)**: $\\frac{FP}{FP + TN}$
    """)

