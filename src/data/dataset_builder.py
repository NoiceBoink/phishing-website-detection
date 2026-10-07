"""
Dataset Builder & Preprocessing Pipeline for Phishing Detection.
Generates and processes representative, high-diversity benign and phishing URLs.
"""

import os
import random
import pandas as pd
from typing import Tuple, List, Optional
from src.features.feature_extractor import URLFeatureExtractor, SUSPICIOUS_TLDS, URL_SHORTENERS, PHISHING_KEYWORDS


BENIGN_DOMAINS = [
    "google.com", "youtube.com", "github.com", "wikipedia.org", "microsoft.com",
    "amazon.com", "apple.com", "linkedin.com", "reddit.com", "stackoverflow.com",
    "twitter.com", "netflix.com", "medium.com", "dropbox.com", "paypal.com",
    "chase.com", "nytimes.com", "bbc.co.uk", "mit.edu", "stanford.edu",
    "harvard.edu", "nih.gov", "nasa.gov", "cnn.com", "theguardian.com",
    "spotify.com", "zoom.us", "cloudflare.com", "salesforce.com", "adobe.com",
    "quora.com", "pinterest.com", "twitch.tv", "tumblr.com", "yahoo.com",
    "duckduckgo.com", "gitlab.com", "archive.org", "kaggle.com", "arxiv.org",
    "coursera.org", "edx.org", "nature.com", "ieee.org", "sciencedirect.com",
    "mozilla.org", "docker.com", "ubuntu.com", "oracle.com", "intel.com"
]

BENIGN_PATHS = [
    "",
    "/",
    "/about",
    "/contact",
    "/privacy-policy",
    "/terms-of-service",
    "/blog/2024/05/machine-learning-advances",
    "/docs/v2/getting-started.html",
    "/wiki/Artificial_intelligence",
    "/projects/open-source-software",
    "/products/enterprise/pricing",
    "/help/faq/account-settings",
    "/news/technology/science-updates",
    "/profile/user?id=104928&tab=activity",
    "/search?q=machine+learning+tutorial&page=2",
    "/watch?v=dQw4w9WgXcQ",
    "/library/archive/research_paper_v3.pdf",
    "/courses/cs101/syllabus",
    "/downloads/releases/version-4.1.2.tar.gz",
    "/api/v1/status?format=json"
]

TARGETED_BRANDS = [
    "paypal", "apple", "microsoft", "google", "netflix", "amazon",
    "chase", "bankofamerica", "wellsfargo", "facebook", "instagram",
    "coinbase", "binance", "metamask", "dhl", "fedex", "usps", "dropbox"
]


def generate_synthetic_dataset(num_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, highly varied benchmark dataset of Benign (0) and Phishing (1) URLs.
    Includes real-world attack vectors: typosquatting, credential harvesting,
    subdomain nesting, IP-based hosting, token obfuscation, and shortened links.
    """
    random.seed(random_state)
    records: List[dict] = []
    half = num_samples // 2

    # --- 1. BENIGN URLs (label = 0) ---
    for _ in range(half):
        domain = random.choice(BENIGN_DOMAINS)
        # Subdomains for benign: occasional 'www', 'mail', 'blog', 'docs', 'api'
        sub = random.choice(["", "www.", "blog.", "docs.", "api.", "m.", "support."])
        full_domain = sub + domain if not domain.startswith("www.") else domain
        path = random.choice(BENIGN_PATHS)
        
        # Optional safe query params
        if path and "?" not in path and random.random() < 0.25:
            path += f"?session_id={random.randint(1000, 99999)}&lang=en"

        # Safe protocol: 90% https, 10% http
        proto = "https://" if random.random() < 0.90 else "http://"
        url = proto + full_domain + path
        records.append({"url": url, "label": 0})

    # --- 2. PHISHING URLs (label = 1) ---
    phish_tlds = list(SUSPICIOUS_TLDS)
    shorteners = list(URL_SHORTENERS)

    for _ in range(num_samples - half):
        brand = random.choice(TARGETED_BRANDS)
        action_keyword = random.choice(PHISHING_KEYWORDS)
        attack_type = random.choice([
            "subdomain_spoof",
            "typosquat",
            "ip_host",
            "suspicious_tld",
            "token_obfuscation",
            "shortened",
            "deep_path_harvesting"
        ])

        proto = "http://" if random.random() < 0.70 else "https://"

        if attack_type == "subdomain_spoof":
            # e.g., http://paypal.com.account-update.xyz/login.php
            fake_tld = random.choice(phish_tlds)
            noise = "".join(random.choices("abcdef0123456789", k=6))
            url = f"{proto}{brand}.com.{action_keyword}-{noise}.{fake_tld}/{action_keyword}"

        elif attack_type == "typosquat":
            # e.g., http://paypa1-security-verification.com/webscr
            typo_brand = brand.replace("o", "0").replace("l", "1").replace("e", "3") if any(c in brand for c in "ole") else brand + "-secure"
            fake_tld = random.choice(["com", "net", "org", "info"] + phish_tlds)
            url = f"{proto}{typo_brand}-{action_keyword}.{fake_tld}/login?auth=true"

        elif attack_type == "ip_host":
            # e.g., http://192.168.1.105:8080/secure/chase/login.php
            ip = f"{random.randint(20, 220)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            port_str = f":{random.choice([8080, 8000, 8888, 3000])}" if random.random() < 0.4 else ""
            url = f"{proto}{ip}{port_str}/{brand}/{action_keyword}.html"

        elif attack_type == "suspicious_tld":
            # e.g., http://netflix-billing-renew.icu/account
            tld = random.choice(phish_tlds)
            url = f"{proto}{brand}-{action_keyword}-center.{tld}/{action_keyword}/verify"

        elif attack_type == "token_obfuscation":
            # e.g., http://secure-portal.com/login?token=24905f019a3b4e9f8e7&redirect=paypal.com
            fake_tld = random.choice(["com", "cc", "top"])
            hex_token = "".join(random.choices("0123456789abcdef", k=32))
            url = f"{proto}{action_keyword}-portal.{fake_tld}/index.php?token={hex_token}&target={brand}"

        elif attack_type == "shortened":
            # e.g., http://bit.ly/3xSecurity
            short = random.choice(shorteners)
            slug = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", k=7))
            url = f"http://{short}/{slug}"

        else: # deep_path_harvesting
            # e.g., http://customer-service-dept.top/webscr/login/cmd/submit?id=823
            tld = random.choice(phish_tlds)
            url = f"{proto}customer-service-{brand}.{tld}/webscr/{action_keyword}/submit?session={random.randint(10000, 99999)}"

        records.append({"url": url, "label": 1})

    # Shuffle dataset
    random.shuffle(records)
    df = pd.DataFrame(records)
    return df


def prepare_dataset(
    output_dir: str = "data",
    num_samples: int = 4000,
    force_rebuild: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds or loads the dataset, extracts 32 features, and saves processed CSVs.
    Returns (raw_df, features_df).
    """
    raw_dir = os.path.join(output_dir, "raw")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    raw_path = os.path.join(raw_dir, "urls.csv")
    features_path = os.path.join(processed_dir, "features_dataset.csv")

    if not force_rebuild and os.path.exists(raw_path) and os.path.exists(features_path):
        print(f"Loading existing cached dataset from {processed_dir}...")
        raw_df = pd.read_csv(raw_path)
        features_df = pd.read_csv(features_path)
        return raw_df, features_df

    print(f"Generating synthetic benchmark dataset of {num_samples} URLs...")
    raw_df = generate_synthetic_dataset(num_samples=num_samples)
    raw_df.to_csv(raw_path, index=False)

    print("Extracting 32 numerical & security features from URLs...")
    extractor = URLFeatureExtractor()
    feature_rows = []

    for idx, row in raw_df.iterrows():
        url = row["url"]
        label = row["label"]
        feats = extractor.extract_features(url)
        feats["label"] = label
        feats["url"] = url
        feature_rows.append(feats)

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(features_path, index=False)
    print(f"Successfully created features dataset with shape: {features_df.shape}")

    return raw_df, features_df


if __name__ == "__main__":
    raw_df, features_df = prepare_dataset(num_samples=4000, force_rebuild=True)
    print("Dataset Distribution:")
    print(raw_df["label"].value_counts())
    print("\nFeature Summary:")
    print(features_df.head(2))

