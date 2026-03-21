# TrueSight — Project Reference

## Overview
TrueSight is a **local, offline AI-powered cyber forensics tool** for detecting AI-generated, manipulated, or suspicious media. It runs entirely on-device using Qwen2 (0.5B) via Ollama.

---

## Architecture Principle

```
Media Input
    │
    ├── 🖼️ Image Module (ViT + ELA + EXIF)
    ├── 🔊 Audio Module (Pitch + MFCC + Spectral)
    ├── 🎥 Video Module (ViT + SSIM + ffprobe + audio)
    └── 🌐 URL Module (Entropy + Homograph + DGA)
                        │
                        ▼
         Numerical Scores (0–100 per modality)
                        │
                        ▼
     ⚙️ Fusion Engine (Weighted Maths — Decision Layer)
        final = 0.35×img + 0.25×aud + 0.25×vid + 0.15×url
                        │
                        ▼
         🧠 Phi-3 Mini (Explanation Layer ONLY)
         → Generates structured forensic narrative
                        │
                        ▼
              📄 PDF Report Download
```

> **Key Rule**: The LLM **never makes the verdict**. The Fusion Engine makes the decision; Qwen2 only writes the explanation.

---

## Module Responsibilities

| File | Role | Techniques |
|---|---|---|
| `app.py` | UI + orchestration | Streamlit |
| `modules/image.py` | Image forensics | ViT (`prithivMLmods/Deep-Fake-Detector-Model`), ELA, EXIF |
| `modules/audio.py` | Audio forensics | Pitch std, MFCC delta, RMS energy, spectral centroid/rolloff/flatness |
| `modules/video.py` | Video forensics | Frame ViT, SSIM consistency, Optical Flow Anomaly, ffprobe metadata |
| `modules/url.py` | URL forensics | Shannon entropy, homograph detection, DGA, shortener heuristics |
| `modules/metadata.py` | EXIF + video tags | exifread, ffprobe JSON |
| `modules/threats.py` | Malware scan | File signature, entropy, extension mismatch |
| `fusion/engine.py` | **Decision Layer** | Weighted formula + confidence computation |
| `llm/llm.py` | **Explanation Layer** | Qwen2 (0.5B) via Ollama |
| `reports/generator.py` | PDF export | reportlab |

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
