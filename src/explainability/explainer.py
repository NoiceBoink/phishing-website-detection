"""
Explainable AI (XAI) Engine for Phishing Detection.
Provides Global Feature Importance and Local Instance-Level Attribution
(Explaining exactly WHY a given URL was flagged as phishing or benign).
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple


# Human-friendly descriptions and safety context for each extracted feature
FEATURE_DESCRIPTIONS = {
    "url_length": "Overall URL character length (unusually long URLs often hide attack payloads)",
    "domain_length": "Domain name length",
    "path_length": "Path length inside the web directory",
    "query_length": "Length of URL query parameters (?param=value)",
    "tld_length": "Top Level Domain length",
    "count_dots": "Number of dots '.' (excessive dots indicate nested deceptive subdomains)",
    "count_hyphens": "Number of hyphens '-' (commonly used to mimic brand names)",
    "count_underscores": "Number of underscores '_'",
    "count_slashes": "Number of directory slashes '/'",
    "count_double_slashes": "Presence of deceptive '//' redirects inside path",
    "count_at": "Presence of '@' symbol (often used to obscure the true host destination)",
    "count_question_marks": "Count of '?' symbols",
    "count_equals": "Count of '=' parameter delimiters",
    "count_percent": "Count of '%' signs indicating URL hex-encoded character obfuscation",
    "count_ampersand": "Count of '&' query parameter separators",
    "count_digits": "Total count of numerical digits in the URL",
    "digit_ratio": "Proportion of digits relative to entire URL length",
    "letter_ratio": "Proportion of letters relative to URL length",
    "special_char_ratio": "Ratio of special characters indicating high symbol density",
    "uppercase_ratio": "Proportion of uppercase letters",
    "url_entropy": "Shannon Entropy of the URL (measures character randomness & DGA generation)",
    "domain_entropy": "Shannon Entropy of the domain name (flags random disposable domains)",
    "has_ip_address": "Direct IP address used instead of a registered domain name (high risk)",
    "subdomain_count": "Number of nested subdomains (e.g., login.chase.com.security.xyz)",
    "has_https": "Use of HTTPS protocol (absence increases vulnerability/risk)",
    "https_in_domain": "Misleading 'https' token embedded in domain name to trick users",
    "suspicious_tld": "High-risk, abusive Top-Level Domain (e.g., .xyz, .top, .icu, .buzz)",
    "is_shortened": "URL shortening service used (hides the final destination)",
    "keyword_count": "Count of credential-harvesting keywords (login, verify, bank, secure)",
    "has_punycode": "Punycode (xn--) detected, indicating potential homoglyph spoofing",
    "has_port": "Non-standard network port specified in URL",
    "has_client_server_token": "Presence of deceptive 'client' or 'server' tokens in domain"
}


class PhishingExplainer:
    """
    Computes global importance rankings and local attribution waterfalls for individual URLs.
    """

    def __init__(self, model, scaler, feature_names: List[str]):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        # Reference benign baseline (medians or zeros)
        self.benign_baseline = np.zeros(len(feature_names))

    def get_global_importance(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Returns the top global features that drive model decisions."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            # Fallback for linear models: absolute coefficients
            importances = np.abs(self.model.coef_[0])
        
        ranked_indices = np.argsort(importances)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            feat = self.feature_names[idx]
            results.append({
                "feature": feat,
                "importance": float(round(importances[idx], 4)),
                "description": FEATURE_DESCRIPTIONS.get(feat, "")
            })
        return results

    def explain_prediction(
        self,
        feature_dict: Dict[str, Any],
        base_probability: float,
        top_k: int = 6
    ) -> Dict[str, Any]:
        """
        Computes local feature contribution (marginal perturbation impact)
        showing which exact features made this URL score high/low risk.
        """
        # Convert feature dict into raw array ordered by self.feature_names
        raw_vals = np.array([feature_dict[col] for col in self.feature_names], dtype=float)
        x_scaled = self.scaler.transform(raw_vals.reshape(1, -1))[0]

        contributions = []

        # Local sensitivity analysis: neutralize each feature to zero (scaled mean)
        # and measure change in phishing probability
        for i, col in enumerate(self.feature_names):
            val = raw_vals[i]
            perturbed = x_scaled.copy()
            perturbed[i] = 0.0 # Neutralize to standardized average

            # Calculate probability after neutralizing feature i
            perturbed_prob = self.model.predict_proba(perturbed.reshape(1, -1))[0, 1]
            # Contribution: how much this feature altered the risk compared to average
            impact = base_probability - perturbed_prob

            contributions.append({
                "feature": col,
                "value": val,
                "impact": float(round(impact, 4)),
                "description": FEATURE_DESCRIPTIONS.get(col, "")
            })

        # Separate into risk-increasing (positive) and risk-decreasing (negative) factors
        positive_factors = [c for c in contributions if c["impact"] > 0.001]
        negative_factors = [c for c in contributions if c["impact"] < -0.001]

        positive_factors.sort(key=lambda x: x["impact"], reverse=True)
        negative_factors.sort(key=lambda x: x["impact"])

        # Top explanations
        top_reasons = []
        for pf in positive_factors[:top_k]:
            reason = self._generate_reason_text(pf["feature"], pf["value"], pf["impact"])
            top_reasons.append(reason)

        return {
            "risk_score_percent": round(base_probability * 100, 2),
            "verdict": "PHISHING" if base_probability >= 0.65 else ("SUSPICIOUS" if base_probability >= 0.35 else "BENIGN"),
            "top_positive_contributors": positive_factors[:top_k],
            "top_mitigating_factors": negative_factors[:top_k],
            "natural_language_reasons": top_reasons
        }

    def _generate_reason_text(self, feat: str, val: float, impact: float) -> str:
        pct = round(impact * 100, 1)
        if feat == "has_ip_address" and val == 1:
            return f"Uses raw IP address instead of domain (+{pct}% risk)"
        elif feat == "suspicious_tld" and val == 1:
            return f"Registered on high-abuse TLD (+{pct}% risk)"
        elif feat == "keyword_count" and val > 0:
            return f"Contains {int(val)} credential/account keywords (+{pct}% risk)"
        elif feat == "subdomain_count" and val > 1:
            return f"Contains {int(val)} nested subdomains (+{pct}% risk)"
        elif feat == "url_entropy":
            return f"High character entropy ({val:.2f} bits) indicating obfuscation (+{pct}% risk)"
        elif feat == "is_shortened" and val == 1:
            return f"Masked via URL shortening service (+{pct}% risk)"
        elif feat == "has_punycode" and val == 1:
            return f"Punycode homoglyph detected (+{pct}% risk)"
        elif feat == "https_in_domain" and val == 1:
            return f"Deceptive 'https' token inside domain string (+{pct}% risk)"
        elif feat == "count_dots" and val > 2:
            return f"High count of dots ({int(val)}) (+{pct}% risk)"
        else:
            return f"Elevated {feat} (value: {val}) contributed +{pct}% to threat score"

