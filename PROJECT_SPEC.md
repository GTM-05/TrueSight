# TrueSight — Project Reference

## Overview
TrueSight is a **local, offline AI-powered cyber forensics tool** for detecting AI-generated, manipulated, or suspicious media. It runs entirely on-device using Qwen2 (0.5B) via Ollama.

---

## 2. Decision Intelligence

### A. Modular Analysis
Each module (Image, Audio, Video, URL) returns a score from **0 to 100**.

### B. Max-Biased Fusion (`fusion/engine.py`)
To prevent dangerous AI signals from being averaged out by clean signals in other modalities, TrueSight uses a **Max-Biased Fusion** logic:

1. **Calculate Weighted Baseline**:
   `weighted_avg = (0.35×Img + 0.25×Aud + 0.25×Vid + 0.15×URL)`
2. **Determine Peak Suspicion**:
   `max_score = max(Img, Aud, Vid, URL, Vid_Flicker, Vid_LipSync)`
3. **Biased Final Score**:
   - If `max_score ≥ 60`: `Final = max_score × 0.9 + (weighted_avg × 0.1)`
   - If `max_score ≥ 30`: `Final = max_score × 0.7 + (weighted_avg × 0.3)`
   - Otherwise: `Final = weighted_avg`

### C. The 19% Safety Floor
If the `Final Score < 75` and no **Strong Forensic Anchors** (FFT Grids, High ViT, or Pulse Anomaly) are found, the score is capped at **19% (Low Risk)**.

---

## 3. Forensic Thresholds (ForensicConfig)

| Constant | Value | Role |
|---|---|---|
| `AI_SYNTH_STRONG` | 45% | Definitive AI probability threshold. |
| `GRID_PEAK_RATIO` | 120 | FFT Spectral Peak sensitivity. |
| `SAFETY_CAP_LIMIT` | 75% | Threshold for applying the noise floor. |
| `SAFETY_CAP_VAL` | 19% | Final low-risk score for noise. |
| `LACK_DATA_PENALTY`| 15% | Penalty for missing biometric data. |

---

## Scoring Thresholds

### Fusion Engine
| Score | Verdict |
|---|---|
| ≥ 60% | 🔴 **High Risk** |
| 30–59% | 🟡 **Medium Risk** |
| < 30% | 🟢 **Low Risk** |

### Confidence Formula
```python
confidence = abs((final_score / 100.0) - 0.5) * 2 * 100
```

### Image Scoring (ViT)
| AI Probability | Points Added |
|---|---|
| ≥ 80% | +80 (CRITICAL) |
| ≥ 50% | +55 (Strong signal) |
| ≥ 35% | +35 (Moderate) |
| ≥ 20% | +15 (Weak) |

### Video Multi-Frame Bonus
| Frames Flagged (≥40%) | Bonus |
|---|---|
| 3+ of 3+ frames | +20 (Consistency Bonus) |
| 2+ of 2+ frames | +10 |

---

## Key Design Decisions

1. **LLM is explanation-only** — prevents hallucinated verdicts
2. **Lazy model loading** — ViT loaded once, cached globally to save RAM
3. **Weight redistribution** — if ffmpeg/ffprobe unavailable, visual evidence takes full weight
4. **tempfile for uploads** — no CWD collision, safe across restarts
5. **Low Resource Mode** — checkbox in sidebar reduces to 1 frame + skips SSIM

---

## Dependencies
```
streamlit         — UI
transformers      — ViT AI image detection
torch             — Model inference backbone
Pillow            — Image loading + ELA
numpy             — Signal math
librosa           — Audio feature extraction
scikit-image      — SSIM computation
opencv-python     — Frame extraction (replaces MoviePy)
exifread          — EXIF metadata parsing
tldextract        — URL domain parsing
reportlab         — PDF generation
ollama            — Qwen2 (0.5B) inference client
```

### System Tools Required
```bash
sudo apt install ffmpeg   # Audio extraction + video metadata (ffprobe included)
ollama pull qwen2:0.5b    # LLM explanation layer (Lite)
```
