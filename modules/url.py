"""
TrueSight — Definitive Master URL Forensics v3.0
modules/url.py
"""

import re
import math
import tldextract
import difflib
from typing import Optional
from config import CFG

def _shannon_entropy(s: str) -> float:
    if not s: return 0.0
    freq = {}
    for c in s: freq[c] = freq.get(c, 0) + 1
    return -sum((f/len(s)) * math.log2(f/len(s)) for f in freq.values())

_LEET_MAP = str.maketrans("013456789", "oieasgtbp")

def _normalize_leet(s: str) -> str:
    """Replace common digit/leet substitutions to catch g00gle → google."""
    return s.lower().translate(_LEET_MAP)

def analyze_url(url: str) -> dict:
    """
    Standard contract v3.0 for URL analysis.
    """
    score = 0
    reasons = []
    features = {}
    is_strong = False
    
    url_lower = url.lower().strip()
    if not url_lower.startswith(('http://', 'https://')):
        url_lower = 'http://' + url_lower

    ext = tldextract.extract(url_lower)
    domain = ext.domain
    tld = ext.suffix
    subdomain = ext.subdomain
    reg_domain = ext.registered_domain

    # Initialize common features for UI
    features.update({
        "domain": domain,
        "tld": tld,
        "subdomain": subdomain,
        "registered_domain": reg_domain,
        "subdomain_depth": len(subdomain.split('.')) if subdomain else 0
    })

    # 1. IP-based URL (exclude localhost + RFC1918 private ranges)
    ip_match = re.search(
        r"https?://(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})", url_lower
    )
    is_ip = bool(ip_match)
    private = False
    if ip_match:
        octets = tuple(int(ip_match.group(i)) for i in range(1, 5))
        a, b, c, d = octets
        if a == 10:
            private = True
        elif a == 172 and 16 <= b <= 31:
            private = True
        elif a == 192 and b == 168:
            private = True
        elif a == 127:
            private = True
    safe_local = any(p in url_lower for p in ["127.0.0.1", "localhost", "::1"])
    if is_ip and (safe_local or private):
        # Private/local IP — return early as Low, no further scoring needed
        return {
            "score": 0.0, "confidence": 0.5, "is_strong": False,
            "risk_level": "Low", "reasons": ["[URL] private/local IP address — not a phishing risk."],
            "features": {**features, "private_ip": True}, "sub_scores": {},
        }
    if is_ip and not safe_local and not private:
        score += CFG.URL_IP_SCORE
        reasons.append("[URL] direct IP address detected — high-confidence phishing anchor.")
        is_strong = True
        features["ip_url"] = True

    # 2. URL shortener
    if reg_domain in CFG.URL_SHORTENERS:
        score += CFG.URL_SHORTENER_SCORE
        reasons.append(f"[URL] shortener detected ({reg_domain}) — obscures destination.")
        is_strong = True
        features['shortener'] = True

    # 3. Suspicious TLD
    if tld in CFG.URL_SUSPICIOUS_TLDS:
        score += CFG.URL_SUSPICIOUS_TLD_SCORE
        reasons.append(f"[URL] suspicious TLD (.{tld}) — heavily abused for phishing.")
        features['suspicious_tld'] = True

    # 4. Homograph attack
    has_latin = bool(re.search(r'[a-zA-Z]', domain))
    has_cyrillic = bool(re.search(r'[\u0400-\u04FF]', domain))
    has_greek = bool(re.search(r'[\u0370-\u03FF]', domain))
    if has_latin and (has_cyrillic or has_greek):
        score += CFG.URL_HOMOGRAPH_SCORE
        reasons.append("[URL] Homograph attack detected (mixed Unicode script).")
        is_strong = True
        features['homograph'] = True

    # 5. Fuzzy Domain Spoofing (includes leet-speak normalization)
    is_golden = False
    domain_norm = _normalize_leet(domain)
    for golden in CFG.URL_GOLDEN_DOMAINS:
        if domain == golden:
            is_golden = True
            break
        # Standard fuzzy match
        ratio = difflib.SequenceMatcher(None, domain, golden).ratio()
        if ratio >= CFG.URL_FUZZY_RATIO_MIN:
            score += CFG.URL_FUZZY_SPOOF_SCORE
            reasons.append(f"[URL] fuzzy domain spoofing: '{domain}' mimics '{golden}'.")
            is_strong = True
            features['fuzzy_spoof'] = golden
            break
        # Leet-speak / digit-substitution check (e.g. g00gle → google)
        ratio_norm = difflib.SequenceMatcher(None, domain_norm, golden).ratio()
        if ratio_norm >= CFG.URL_FUZZY_RATIO_MIN and domain_norm != domain:
            score += CFG.URL_FUZZY_SPOOF_SCORE
            reasons.append(f"[URL] leet-speak domain spoofing: '{domain}' mimics '{golden}' (digit substitution).")
            is_strong = True
            features['fuzzy_spoof'] = golden
            break

    # 6. HTTP Only
    if not url_lower.startswith('https://') and not is_golden:
        # Don't penalize local dev stuff too much
        if not any(p in url_lower for p in ["localhost", "127.0.0.1"]):
            score += CFG.URL_NO_HTTPS_SCORE
            reasons.append("[URL] HTTP only — no transport encryption.")

    # 7. Phishing keywords (golden domains are exempt — google.com/microsoftonline.com etc.)
    if not is_golden:
        full_dom = f"{subdomain}.{domain}" if subdomain else domain
        full_dom = full_dom.lower()
        found_kws = [kw for kw in CFG.URL_PHISHING_KEYWORDS if kw in full_dom]
        if found_kws:
            k_score = min(CFG.URL_KEYWORD_MAX, len(found_kws) * CFG.URL_KEYWORD_SCORE_PER)
            score += k_score
            reasons.append(f"[URL] phishing keywords detected: {', '.join(found_kws[:3])}. ")


    # 8. Entropy
    ent = _shannon_entropy(domain)
    features["domain_entropy"] = round(ent, 2)
    if ent > CFG.URL_ENTROPY_HIGH:
        score += CFG.URL_ENTROPY_HIGH_SCORE
        reasons.append(f"[URL] high domain entropy ({ent:.2f}) — likely random DGA.")
    elif ent > CFG.URL_ENTROPY_MED:
        score += CFG.URL_ENTROPY_MED_SCORE
        reasons.append(f"[URL] elevated domain entropy ({ent:.2f}).")

    # 9. Subdomain nesting
    depth = len(subdomain.split('.')) if subdomain else 0
    if depth >= 3:
        score += CFG.URL_SUBDOMAIN_DEEP_SCORE
        reasons.append(f"[URL] deep subdomain nesting ({depth} levels).")
    elif depth == 2:
        score += CFG.URL_SUBDOMAIN_MED_SCORE
        reasons.append(f"[URL] multiple subdomain levels.")

    # 10. Length
    u_len = len(url)
    if u_len > CFG.URL_LENGTH_LONG:
        score += CFG.URL_LENGTH_LONG_SCORE
        reasons.append(f"[URL] excessively long URL ({u_len} chars).")
    elif u_len > CFG.URL_LENGTH_MED:
        score += CFG.URL_LENGTH_MED_SCORE
        reasons.append(f"[URL] long URL ({u_len} chars).")

    # 11. Digit ratio + Redirect params
    digit_ratio = sum(c.isdigit() for c in domain) / (len(domain) or 1)
    if digit_ratio > CFG.URL_DIGIT_RATIO_MAX:
        score += CFG.URL_DIGIT_SCORE
        reasons.append(f"[URL] high digit ratio in domain ({digit_ratio:.1%}).")

    found_params = [p for p in CFG.URL_REDIRECT_PARAMS if p in url_lower]
    if found_params:
        score += CFG.URL_REDIRECT_PARAM_SCORE
        reasons.append(f"[URL] suspicious redirect parameters: {', '.join(found_params[:2])}.")

    final_score = float(min(100, score))
    conf_out = 0.90
    if features.get("homograph"):
        conf_out = max(conf_out, 0.96)
    if features.get("ip_url"):
        conf_out = max(conf_out, 0.99)
    if features.get("shortener"):
        conf_out = max(conf_out, 0.95)
    if features.get("fuzzy_spoof"):
        conf_out = max(conf_out, 0.91)

    return {
        "score":      final_score,
        "confidence": conf_out,
        "is_strong":  is_strong,
        "risk_level": "High" if final_score >= CFG.HIGH_RISK_THRESHOLD else "Medium" if final_score >= CFG.MEDIUM_RISK_THRESHOLD else "Low",
        "reasons":    reasons,
        "features":   features,
        "sub_scores": {},
    }
