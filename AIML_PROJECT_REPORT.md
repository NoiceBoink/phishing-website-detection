# ACADEMIC PROJECT REPORT: PHISHING WEBSITE DETECTION SYSTEM
**Domain**: Artificial Intelligence, Machine Learning & Cybersecurity  
**System Name**: PhishShield AI  

---

## 1. ABSTRACT
Phishing represents one of the most prevalent and damaging cyber threats, utilizing deceptive URLs and social engineering to steal user credentials, financial information, and sensitive data. Traditional blacklisting methods fail against zero-hour attacks, dynamic domain generation algorithms (DGA), and polymorphic URL structures. This project proposes a dual-paradigm detection framework that integrates 32 handcrafted lexical, structural, and statistical security features with an end-to-end Character-Level Deep Learning model (1D-CNN + Bidirectional LSTM). The system benchmarks multiple algorithms (Random Forest, HistGradientBoosting, Extra Trees, and Logistic Regression), incorporates Explainable AI (XAI) for transparent threat diagnostics, and delivers results via an interactive Streamlit dashboard and a FastAPI REST service.

---

## 2. PROBLEM STATEMENT
Phishing URLs are engineered to deceive human users and traditional static filters by employing:
1. **Typosquatting & Homoglyphs**: Replacing characters with visually indistinguishable Unicode counterparts (e.g., Cyrillic 'а' vs Latin 'a').
2. **Subdomain Nesting**: Embedding legitimate brand strings into deep subdomain paths (`paypal.com.verification-security.xyz`).
3. **Obfuscation & DGA**: Generating high-entropy randomized URL paths or utilizing URL shortening services to mask malicious destinations.

**Objective**: To build an automated, explainable, and real-time AI/ML classification pipeline that predicts the legitimacy of any given URL with low inference latency and high precision/recall.

---

## 3. METHODOLOGY & MATHEMATICAL FORMULATIONS

### 3.1 Feature Engineering (32 Handcrafted Features)
The system categorizes URL characteristics into four distinct groups:

1. **Lexical & Length Features**:
   - $L_{\text{url}}$: Total URL string length.
   - $L_{\text{domain}}$: Length of the hostname.
   - $L_{\text{path}}$: Length of the resource path.
   - $L_{\text{query}}$: Length of the query string.
   - $L_{\text{tld}}$: Top-level domain length.

2. **Syntactic Delimiters & Character Distributions**:
   - Counts of critical delimiters: `.`, `-`, `_`, `/`, `//`, `@`, `?`, `=`, `%`, `&`.
   - Digits count, digit-to-length ratio, letter ratio, and uppercase ratio.

3. **Information Theory & Statistical Randomness**:
   - **Shannon Entropy ($H$)**: Measures the uncertainty and randomness in the URL string to detect algorithmic domain generation (DGA) and obfuscated payloads:
     $$H(S) = - \sum_{i=1}^{k} P(c_i) \log_2 P(c_i)$$
     where $P(c_i) = \frac{\text{count}(c_i)}{|S|}$ for each unique character $c_i$ in string $S$.
   - Computed for both the full URL ($H_{\text{url}}$) and the domain segment ($H_{\text{domain}}$).

4. **Security & Semantic Heuristics**:
   - Binary indicator for direct IPv4/IPv6 address as hostname ($\mathbb{I}_{\text{ip}}$).
   - Subdomain hierarchy depth ($N_{\text{sub}}$).
   - Protocol security ($\mathbb{I}_{\text{https}}$) and deceptive HTTPS token inside domain string ($\mathbb{I}_{\text{fake\_https}}$).
   - High-abuse TLD flag ($\mathbb{I}_{\text{bad\_tld}}$ for `.xyz`, `.top`, `.icu`, `.buzz`, etc.).
   - URL shortener mask detection ($\mathbb{I}_{\text{short}}$).
   - Credential-harvesting keywords count ($K_{\text{phish}}$ for terms such as `login`, `verify`, `bank`, `secure`, `webscr`, `update`).
   - Punycode homoglyph check ($\mathbb{I}_{\text{punycode}}$ for `xn--`).
   - Non-standard port detection ($\mathbb{I}_{\text{port}}$).

---

### 3.2 Machine Learning Model Architectures
1. **Random Forest Classifier**:
   - Ensemble of $B = 120$ decorrelated decision trees using bootstrap aggregating (bagging).
   - Feature splits chosen via Gini Impurity:
     $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
2. **Histogram-based Gradient Boosting (HistGradientBoosting)**:
   - Builds trees sequentially to minimize the binary cross-entropy loss function:
     $$\mathcal{L}(y, \hat{y}) = - \left[ y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \right]$$
   - Discretizes continuous numerical features into integer-valued bins for fast training.
3. **Extra Trees Classifier (Extremely Randomized Trees)**:
   - Randomizes both feature subsets and split thresholds, offering lower variance.
4. **Logistic Regression**:
   - Linear baseline utilizing the sigmoid activation:
     $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

---

### 3.3 Deep Learning Architecture (Char-CNN-BiLSTM)
To evaluate whether manual feature engineering is necessary, an end-to-end neural network processes raw character sequences without manual feature extraction:

1. **Character Vocabulary Embedding**:
   - 128 printable ASCII characters mapped into a 64-dimensional dense vector space:
     $$E \in \mathbb{R}^{L \times 64}, \quad L = 200$$
2. **1D Convolutional Neural Network (Local Pattern Extraction)**:
   - Kernel size $k=5$, filters $F=128$, stride $1$:
     $$C_t = \text{ReLU}\left( W_c * E_{t:t+k-1} + b_c \right)$$
   - Captures character n-grams corresponding to substrings, keywords, and domain tokens.
3. **Bidirectional LSTM (Sequential Context)**:
   - Processes features in both forward and backward directions to preserve contextual order:
     $$\vec{h}_t = \text{LSTM}(C_t, \vec{h}_{t-1}), \quad \overleftarrow{h}_t = \text{LSTM}(C_t, \overleftarrow{h}_{t+1})$$
     $$h_t = [\vec{h}_t \,\|\, \overleftarrow{h}_t]$$
4. **Global Max Pooling & Dense Classification**:
   - Max-pooled over sequence length followed by dense projection to a single sigmoid output.

---

### 3.4 Explainable AI (XAI) Engine
To prevent black-box opacity in security decisions, the system implements:
1. **Global Feature Importance**: Gini importance rankings identifying the overall most discriminative security indicators.
2. **Local Perturbation Sensitivity Attribution**:
   - For a given URL with predicted risk $P_0$, each feature $f_i$ is perturbed to the standardized population baseline:
     $$\Delta \text{Risk}(f_i) = P_0 - P(\mathbf{x}_{-f_i})$$
   - Produces an interpretable waterfall of positive threat factors and mitigating safety factors.

---

## 4. EVALUATION METRICS
In cybersecurity applications, traditional accuracy is insufficient due to operational costs of false classifications:
- **Precision**: $\frac{TP}{TP + FP}$ — Low precision causes false alarms, blocking users from legitimate websites.
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — Low recall allows malicious phishing sites to compromise users.
- **F1-Score**: Harmonic mean of Precision and Recall:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **ROC-AUC**: Area Under the Receiver Operating Characteristic curve.
- **False Positive Rate (FPR)**: $\frac{FP}{FP + TN}$
- **False Negative Rate (FNR)**: $\frac{FN}{FN + TP}$
- **Inference Latency**: Milliseconds per URL classification (essential for real-time web browsing safety).

---

## 5. VIVA VOCE / PROJECT DEFENSE Q&A GUIDE

### Q1: Why did you choose URL-based detection over webpage content scraping?
**Answer**: Webpage content scraping (DOM analysis, screenshots, OCR) requires fetching the page, executing JavaScript, and downloading assets. This introduces latency (2–5 seconds per page) and exposes the scanner to drive-by download exploits. URL-based lexical and statistical analysis operates in milliseconds (< 5 ms) before the user's browser ever establishes a connection to the malicious server.

### Q2: Why is Shannon Entropy important in phishing detection?
**Answer**: Shannon Entropy quantifies character randomness. Legitimate domains generally follow human-readable language patterns (low entropy, e.g., 2.5–3.2 bits). Attackers using Domain Generation Algorithms (DGA), disposable subdomains, or encrypted/obfuscated parameter strings exhibit significantly higher entropy (> 4.2 bits), making it a strong predictive feature.

### Q3: What is the advantage of comparing Classical ML with Deep Learning?
**Answer**: Classical ML with handcrafted features provides high interpretability, fast training, and low latency. Deep Learning (Char-CNN-BiLSTM) operates directly on raw strings without manual feature design. Comparing both provides an ablation study demonstrating whether domain-specific feature engineering outperforms raw sequence representation learning.

### Q4: Why is Precision considered as critical as Recall in this domain?
**Answer**: While missing a phishing site (False Negative) leads to credential theft, misidentifying a legitimate institutional or banking website as phishing (False Positive) disrupts critical business workflows and causes user alert fatigue, leading users to disable security warnings altogether.

### Q5: How does your system address homoglyph attacks?
**Answer**: Homoglyph attacks employ lookalike characters from international alphabets (e.g., Cyrillic 'а' replacing Latin 'a'). Browsers translate these to Punycode (`xn--...`). Our feature extractor explicitly checks for the `xn--` prefix and abnormal character encodings.

### Q6: How does the Explainable AI (XAI) component work?
**Answer**: We implement local perturbation sensitivity. For any scanned URL, the model calculates the baseline threat probability. It then neutralizes each feature to its mean baseline and computes the marginal difference ($\Delta \text{Risk}$). Features that cause the greatest drop in risk are flagged as the primary threat contributors.

