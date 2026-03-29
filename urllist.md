# TrueSight — URL Test List
> Last updated: 2026-03-29 | Accuracy: **100%** (8/8 fake detected, 8/8 legit clean)

This file documents all URLs used to validate the TrueSight URL analysis module.
Use `timeout 30 ./venv/bin/python3 /tmp/test_url_full.py` to re-run tests.

---

## 🔴 Fake / Phishing URLs

| URL | Expected Risk | Detection Trigger |
|-----|:---:|---|
| `http://paypa1-secure.login.ml` | **High** | Leet-speak paypal (`paypa1`) + suspicious TLD (`.ml`) |
| `http://185.220.101.42/fb/login.php` | **High** | Raw public IP address — high-confidence phishing anchor |
| `https://apple-id-suspended.com/verify` | **Medium** | Phishing keywords: `apple`, `suspended` |
| `http://amaz0n-account-update.tk` | **High** | Leet-speak amazon (`amaz0n`) + suspicious TLD (`.tk`) |
| `https://secure.netflix-billing.ml/pay` | **High** | Spoof of netflix + suspicious TLD (`.ml`) |
| `http://bit.ly/3xYKq9P` | **Medium** | URL shortener — obscures real destination |
| `https://signin.micro-soft365.com/login` | **High** | Fuzzy domain spoof of `microsoft` + login keyword |
| `http://rU9xPq3mzLkW.top/cmd` | **Medium** | DGA-like random domain + suspicious TLD (`.top`) |

---

## 🟢 Legit / Real URLs

| URL | Expected Risk | Notes |
|-----|:---:|---|
| `https://www.paypal.com/signin` | **Low** | Official PayPal — golden domain |
| `https://appleid.apple.com/sign-in` | **Low** | Official Apple ID — golden domain |
| `https://www.amazon.com` | **Low** | Official Amazon — golden domain |
| `https://netflix.com` | **Low** | Official Netflix — golden domain |
| `https://login.microsoftonline.com` | **Low** | Official Microsoft Online — golden domain |
| `https://github.com` | **Low** | Official GitHub — golden domain |
| `https://www.linkedin.com/login` | **Low** | Official LinkedIn — golden domain |
| `https://dropbox.com` | **Low** | Official Dropbox — golden domain |

---

## 🔬 Edge Cases Tested

| URL | Risk | Notes |
|-----|:---:|---|
| `http://192.168.1.1` | **Low** | Private IP — safe local network, early exit |
| `http://8.8.8.8` | **High** | Public IP — direct phishing anchor |
| `http://g00gle.com` | **High** | Leet-speak digit substitution (`0→o`) |
| `https://www.google.com` | **Low** | Golden domain — keyword check exempt |
| `http://secure-amazon-login.xyz` | **High** | Keywords + suspicious TLD (`.xyz`) |
| `http://127.0.0.1` | **Low** | Localhost — safe, early exit |

---

## 📋 Detection Signals Reference

| Signal | Config Key | Score |
|--------|-----------|------:|
| Public IP address | `URL_IP_SCORE` | +40 |
| URL shortener | `URL_SHORTENER_SCORE` | +35 |
| Suspicious TLD (`.ml`, `.tk`, `.xyz`…) | `URL_SUSPICIOUS_TLD_SCORE` | +25 |
| Homograph attack (mixed Unicode) | `URL_HOMOGRAPH_SCORE` | +45 |
| Fuzzy domain spoof | `URL_FUZZY_SPOOF_SCORE` | +60 |
| Leet-speak spoof (`g00gle`) | `URL_FUZZY_SPOOF_SCORE` | +60 |
| HTTP only (no TLS) | `URL_NO_HTTPS_SCORE` | +15 |
| Phishing keyword (per keyword) | `URL_KEYWORD_SCORE_PER` | +20 |
| High domain entropy (DGA) | `URL_ENTROPY_HIGH_SCORE` | +20 |
| Deep subdomain nesting | `URL_SUBDOMAIN_DEEP_SCORE` | +20 |
| Suspicious redirect params | `URL_REDIRECT_PARAM_SCORE` | +15 |

**Risk Thresholds:** High ≥ 60% · Medium ≥ 30% · Low < 30%
