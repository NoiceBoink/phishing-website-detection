"""
Feature Extractor for Phishing Website Detection System.
Extracts 32 comprehensive lexical, syntactic, statistical, and security heuristic features
from raw URL strings.
"""

import math
import re
from urllib.parse import urlparse
from typing import Dict, Any, List


SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "work", "buzz", "tk", "ml", "ga", "cf", "gq", 
    "fit", "rest", "icu", "cam", "kim", "country", "science", "gdn", "stream",
    "men", "win", "bid", "loan", "date", "racing", "download", "accountant",
    "faith", "cricket", "party", "review", "trade", "webcam", "space"
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly", 
    "goo.gl", "bit.do", "cutt.ly", "rb.gy", "tiny.cc", "shorte.st", "bc.vc"
}

PHISHING_KEYWORDS = [
    "login", "signin", "verify", "verification", "account", "update", "banking",
    "secure", "security", "webscr", "ebayisapi", "password", "confirm", "auth",
    "authentication", "support", "billing", "wallet", "recover", "unlock", "alert",
    "service", "portal", "pay", "payment", "invoice", "validate", "suspended"
]


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of a given string."""
    if not text:
        return 0.0
    entropy = 0.0
    text_len = len(text)
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    for count in counts.values():
        prob = count / text_len
        entropy -= prob * math.log2(prob)
    return float(round(entropy, 4))


def is_ip_address(domain: str) -> int:
    """Check if the domain string is an IPv4 or IPv6 address."""
    # Match standard IPv4
    ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    # Match IPv6
    ipv6_pattern = r"^\[?([0-9a-fA-F]{0,4}:){1,7}[0-9a-fA-F]{0,4}\]?$"
    if re.match(ipv4_pattern, domain):
        # Validate octet ranges
        parts = domain.split(".")
        if all(0 <= int(p) <= 255 for p in parts if p.isdigit()):
            return 1
    if re.match(ipv6_pattern, domain):
        return 1
    return 0


class URLFeatureExtractor:
    """
    Extracts numerical features from a URL for machine learning classification.
    """

    FEATURE_NAMES: List[str] = [
        "url_length",
        "domain_length",
        "path_length",
        "query_length",
        "tld_length",
        "count_dots",
        "count_hyphens",
        "count_underscores",
        "count_slashes",
        "count_double_slashes",
        "count_at",
        "count_question_marks",
        "count_equals",
        "count_percent",
        "count_ampersand",
        "count_digits",
        "digit_ratio",
        "letter_ratio",
        "special_char_ratio",
        "uppercase_ratio",
        "url_entropy",
        "domain_entropy",
        "has_ip_address",
        "subdomain_count",
        "has_https",
        "https_in_domain",
        "suspicious_tld",
        "is_shortened",
        "keyword_count",
        "has_punycode",
        "has_port",
        "has_client_server_token"
    ]

    def __init__(self):
        pass

    def extract_features(self, url: str) -> Dict[str, Any]:
        """
        Parses a URL string and extracts all 32 numerical features.
        """
        raw_url = str(url).strip()
        
        # Ensure scheme is present for proper urlparse behavior
        if not raw_url.startswith(("http://", "https://")):
            parsed = urlparse("http://" + raw_url)
            original_has_scheme = False
        else:
            parsed = urlparse(raw_url)
            original_has_scheme = True

        netloc = parsed.netloc.lower()
        path = parsed.path
        query = parsed.query

        # Strip standard port if present
        domain_without_port = netloc.split(":")[0] if ":" in netloc else netloc

        # Extract domain tokens & TLD
        domain_parts = domain_without_port.split(".")
        tld = domain_parts[-1] if len(domain_parts) > 1 else ""

        # 1. Length features
        url_length = len(raw_url)
        domain_length = len(netloc)
        path_length = len(path)
        query_length = len(query)
        tld_length = len(tld)

        # 2. Character counts
        count_dots = raw_url.count(".")
        count_hyphens = raw_url.count("-")
        count_underscores = raw_url.count("_")
        count_slashes = raw_url.count("/")
        # Count double slashes inside the path (after the initial scheme://)
        double_slash_pos = raw_url.find("//")
        if double_slash_pos != -1:
            after_scheme = raw_url[double_slash_pos + 2:]
            count_double_slashes = after_scheme.count("//")
        else:
            count_double_slashes = 0

        count_at = raw_url.count("@")
        count_question_marks = raw_url.count("?")
        count_equals = raw_url.count("=")
        count_percent = raw_url.count("%")
        count_ampersand = raw_url.count("&")
        
        count_digits = sum(c.isdigit() for c in raw_url)
        count_letters = sum(c.isalpha() for c in raw_url)
        count_special = url_length - (count_digits + count_letters)
        count_uppercase = sum(c.isupper() for c in raw_url)

        # Ratios
        safe_len = max(url_length, 1)
        digit_ratio = round(count_digits / safe_len, 4)
        letter_ratio = round(count_letters / safe_len, 4)
        special_char_ratio = round(count_special / safe_len, 4)
        uppercase_ratio = round(count_uppercase / safe_len, 4)

        # 3. Statistical / Entropy features
        url_entropy = calculate_entropy(raw_url)
        domain_entropy = calculate_entropy(domain_without_port)

        # 4. Security & Semantic Heuristics
        has_ip = is_ip_address(domain_without_port)

        # Subdomain count (e.g. foo.bar.example.com -> 2 subdomains)
        if has_ip:
            subdomain_count = 0
        else:
            # subtract 2 for domain name + tld
            subdomain_count = max(0, len(domain_parts) - 2)

        has_https = 1 if (original_has_scheme and raw_url.lower().startswith("https://")) else 0
        https_in_domain = 1 if ("https" in domain_without_port or "http" in domain_without_port and not domain_without_port.startswith("http")) else 0

        suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0
        is_shortened = 1 if any(short in netloc for short in URL_SHORTENERS) else 0

        # Phishing keywords detection
        url_lower = raw_url.lower()
        keyword_count = sum(1 for kw in PHISHING_KEYWORDS if kw in url_lower)

        has_punycode = 1 if "xn--" in domain_without_port else 0
        has_port = 1 if (":" in netloc and not netloc.startswith("[")) else 0
        
        has_client_server_token = 1 if ("client" in domain_without_port or "server" in domain_without_port) else 0

        return {
            "url_length": url_length,
            "domain_length": domain_length,
            "path_length": path_length,
            "query_length": query_length,
            "tld_length": tld_length,
            "count_dots": count_dots,
            "count_hyphens": count_hyphens,
            "count_underscores": count_underscores,
            "count_slashes": count_slashes,
            "count_double_slashes": count_double_slashes,
            "count_at": count_at,
            "count_question_marks": count_question_marks,
            "count_equals": count_equals,
            "count_percent": count_percent,
            "count_ampersand": count_ampersand,
            "count_digits": count_digits,
            "digit_ratio": digit_ratio,
            "letter_ratio": letter_ratio,
            "special_char_ratio": special_char_ratio,
            "uppercase_ratio": uppercase_ratio,
            "url_entropy": url_entropy,
            "domain_entropy": domain_entropy,
            "has_ip_address": has_ip,
            "subdomain_count": subdomain_count,
            "has_https": has_https,
            "https_in_domain": https_in_domain,
            "suspicious_tld": suspicious_tld,
            "is_shortened": is_shortened,
            "keyword_count": keyword_count,
            "has_punycode": has_punycode,
            "has_port": has_port,
            "has_client_server_token": has_client_server_token
        }

