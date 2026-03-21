import re
import math
from urllib.parse import urlparse
import tldextract

SUSPICIOUS_KEYWORDS = ['login', 'verify', 'bank', 'update', 'secure', 'account', 'auth', 'signin', 'password', 'service']

def extract_features(url):
    """Extract lexical features from a URL for ML and heuristic analysis."""
    if not url.startswith('http'):
        url_parse_target = 'http://' + url
    else:
        url_parse_target = url
        
    parsed = urlparse(url_parse_target)
    ext = tldextract.extract(url_parse_target)
    
    domain = ext.domain
    subdomain = ext.subdomain
    
    # Heuristics
    length = len(url)
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_at = url.count('@')
    is_ip = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) else 0
    
    # Calculate entropy of domain
    prob = [float(domain.count(c)) / len(domain) for c in dict.fromkeys(list(domain))] if domain else []
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob]) if prob else 0.0
    
    # Check suspicious keywords
    found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()]
    
    return {
        'length': length,
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_at': num_at,
        'is_ip': is_ip,
        'entropy': entropy,
        'keyword_count': len(found_keywords),
        'found_keywords': found_keywords,
        'domain': domain,
        'subdomains': subdomain.split('.') if subdomain else []
    }

def analyze_url(url):
    """
    Combines rule-based heuristics to generate a phishing risk score (0-100%).
    Suitable for offline use without external ML APIs.
    """
    if not url:
        return {'score': 0, 'reasons': ['No URL provided'], 'risk_level': 'Low'}
        
    features = extract_features(url)
    
    score = 0
    reasons = []
    
    if features['is_ip']:
        score += 40
        reasons.append("Use of IP address instead of domain name")
        
    if features['num_at'] > 0:
        score += 30
        reasons.append("Contains '@' symbol (often used to obscure URLs)")
        
    if features['keyword_count'] > 0:
        score += 20 * features['keyword_count']
        reasons.append(f"Contains suspicious keywords: {', '.join(features['found_keywords'])}")
        
    if len(features['subdomains']) >= 2:
        score += 15
        reasons.append(f"Multiple subdomains ({len(features['subdomains'])})")
        
    if features['num_dots'] >= 3:
        score += 15
        reasons.append(f"Unusually high number of dots ({features['num_dots']})")
        
    if features['num_hyphens'] >= 2:
        score += 10
        reasons.append("Multiple hyphens in domain")
        
    if features['entropy'] > 4.0:
        score += 15
        reasons.append(f"High domain entropy ({features['entropy']:.2f})")
        
    if features['length'] > 75:
        score += 10
        reasons.append("Unusually long URL")
        
    final_score = min(100, score)
    
    return {
        'score': final_score,
        'reasons': reasons,
        'features': features,
        'risk_level': 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'
    }
