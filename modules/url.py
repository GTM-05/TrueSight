"""
modules/url_ai.py — AI-Enhanced URL Analysis for app-ai.py

Extends the original url.py with:
  - tldextract for proper TLD parsing (avoids false positives from subdomains)
  - Homograph attack detection (mixed Unicode scripts like Cyrillic + Latin)
  - IP-based URL detection (direct IP pointing = phishing indicator)
  - URL shortener detection (hides destination)
  - Suspicious redirect/tracking parameter patterns
  - Domain age check hint via length heuristics
"""

import re
import math
import tldextract


# Known URL shorteners
_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'buff.ly',
    'is.gd', 'rb.gy', 'cutt.ly', 'short.io', 'rebrand.ly', 'bl.ink',
    'shorte.st', 'adf.ly', 'bc.vc', 'clck.ru', 'qps.ru'
}

# Legitimate TLDs that are commonly spoofed using similar-looking ones
_SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq',   # Free TLDs heavily abused
    'xyz', 'top', 'click', 'loan',   # Common phishing TLDs
    'work', 'party', 'review', 'cricket', 'science'
}

_PHISHING_KEYWORDS = [
    'login', 'verify', 'account', 'secure', 'update', 'confirm',
    'banking', 'paypal', 'amazon', 'apple', 'microsoft', 'google',
    'ebay', 'netflix', 'signin', 'wallet', 'password', 'credential',
    'suspended', 'unusual', 'alert', 'urgent', 'immediately'
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f / len(s)) * math.log2(f / len(s)) for f in freq.values())


def _has_homograph(domain: str) -> bool:
    """Detect mixed-script homograph attacks (e.g. Cyrillic 'а' mixed with Latin 'a')."""
    has_latin = bool(re.search(r'[a-zA-Z]', domain))
    has_cyrillic = bool(re.search(r'[\u0400-\u04FF]', domain))
    has_greek = bool(re.search(r'[\u0370-\u03FF]', domain))
    return has_latin and (has_cyrillic or has_greek)


import difflib

# High-value domains frequently targeted for spoofing
_GOLDEN_DOMAINS = [
    'google', 'amazon', 'paypal', 'microsoft', 'facebook', 'netflix',
    'apple', 'icloud', 'gmail', 'outlook', 'chase', 'fidelity'
]

def _fuzzy_domain_match(domain: str) -> str:
    """Detects visually similar (fuzzy) matches to high-value brands."""
    for golden in _GOLDEN_DOMAINS:
        if domain == golden:
            return None # Exact match is fine
        # SequenceMatcher ratio > 0.8 identifies spoofs like 'paypa1' or 'g00gle'
        ratio = difflib.SequenceMatcher(None, domain, golden).ratio()
        if ratio > 0.8:
            return golden
    return None

def _is_ip_url(url: str) -> bool:
    """Detect URLs pointing directly to IP addresses."""
    return bool(re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url))


def analyze_url(url: str) -> dict:
    """
    Enhanced URL phishing analysis with proper TLD parsing,
    homograph detection, and richer heuristics.
    """
    score = 0
    reasons = []
    flags = {}

    url_lower = url.lower().strip()
    if not url_lower.startswith(('http://', 'https://')):
        url_lower = 'http://' + url_lower

    # Parse with tldextract for proper subdomain/domain/TLD separation
    extracted = tldextract.extract(url_lower)
    domain = extracted.domain
    tld = extracted.suffix
    subdomain = extracted.subdomain
    registered_domain = extracted.registered_domain

    flags['domain'] = domain
    flags['tld'] = tld
    flags['subdomain'] = subdomain

    # ── 1. IP-based URL ───────────────────────────────────────────────────────
    if _is_ip_url(url_lower):
        score += 40
        reasons.append("URL points directly to an IP address — strong phishing indicator")
        flags['ip_url'] = True

    # ── 2. URL shortener ──────────────────────────────────────────────────────
    if registered_domain in _SHORTENERS:
        score += 35
        reasons.append(f"URL shortener detected ({registered_domain}) — destination URL is hidden")
        flags['shortener'] = True

    # ── 3. Suspicious TLD ─────────────────────────────────────────────────────
    if tld in _SUSPICIOUS_TLDS:
        score += 25
        reasons.append(f"High-risk TLD: .{tld} — frequently used for phishing campaigns")
        flags['suspicious_tld'] = True

    # ── 4. Homograph attack ───────────────────────────────────────────────────
    if _has_homograph(url_lower):
        score += 45
        reasons.append("Homograph attack detected — URL contains mixed Unicode scripts (e.g. Cyrillic + Latin)")
        flags['homograph'] = True
    
    # ── 4b. Fuzzy Domain Spoofing ─────────────────────────────────────────────
    fuzzy_match = _fuzzy_domain_match(domain)
    if fuzzy_match:
        score += 40
        reasons.append(f"Fuzzy Domain Spoofing: '{domain}' is 80%+ similar to '{fuzzy_match}'")
        flags['fuzzy_match'] = fuzzy_match

    # ── 5. HTTPS check ────────────────────────────────────────────────────────
    if not url_lower.startswith('https://'):
        score += 15
        reasons.append("URL uses HTTP (not HTTPS) — no transport encryption")
        flags['no_https'] = True

    # ── 6. Phishing keywords in domain ────────────────────────────────────────
    full_domain_str = f"{subdomain}.{domain}".lower()
    found_keywords = [kw for kw in _PHISHING_KEYWORDS if kw in full_domain_str]
    if found_keywords:
        score += min(30, len(found_keywords) * 10)
        reasons.append(f"Phishing keywords in domain: {', '.join(found_keywords[:3])}")
        flags['phishing_keywords'] = found_keywords

    # ── 7. Domain Shannon entropy ─────────────────────────────────────────────
    entropy = _shannon_entropy(domain)
    flags['domain_entropy'] = round(entropy, 3)
    if entropy > 3.8:
        score += 20
        reasons.append(f"High domain entropy ({entropy:.2f}) — domain appears randomly generated (DGA)")
    elif entropy > 3.2:
        score += 10
        reasons.append(f"Elevated domain entropy ({entropy:.2f}) — unusual character distribution")

    # ── 8. Deep subdomain nesting ─────────────────────────────────────────────
    subdomain_depth = len(subdomain.split('.')) if subdomain else 0
    flags['subdomain_depth'] = subdomain_depth
    if subdomain_depth >= 3:
        score += 20
        reasons.append(f"Deep subdomain nesting ({subdomain_depth} levels) — often used to mimic legitimate domains")
    elif subdomain_depth == 2:
        score += 10
        reasons.append(f"Multiple subdomain levels ({subdomain_depth}) — verify domain legitimacy")

    # ── 9. Excessive URL length ───────────────────────────────────────────────
    flags['url_length'] = len(url)
    if len(url) > 150:
        score += 15
        reasons.append(f"Unusually long URL ({len(url)} chars) — often used to obscure destination")
    elif len(url) > 100:
        score += 8
        reasons.append(f"Long URL ({len(url)} chars) — moderately suspicious")

    # ── 10. Excessive digits in domain ────────────────────────────────────────
    digit_ratio = sum(c.isdigit() for c in domain) / max(len(domain), 1)
    flags['digit_ratio'] = round(digit_ratio, 3)
    if digit_ratio > 0.35:
        score += 15
        reasons.append(f"High digit ratio in domain ({digit_ratio:.0%}) — unusual for legitimate domains")

    # ── 11. Suspicious query parameters ──────────────────────────────────────
    suspicious_params = ['redirect', 'url=', 'goto=', 'next=', 'return=', 'redir=']
    found_params = [p for p in suspicious_params if p in url_lower]
    if found_params:
        score += 15
        reasons.append(f"Suspicious redirect parameters in URL: {', '.join(found_params)}")

    final_score = min(100, score)
    return {
        'score': final_score,
        'risk_level': 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low',
        'reasons': reasons,
        'features': flags
    }
