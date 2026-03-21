# TrueSight — System Architecture

## Pipeline Overview

```
                    ┌─────────────────────────┐
                    │        app.py           │
                    │   (Streamlit UI + Tabs) │
                    └────────────┬────────────┘
                                 │ upload
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌───────────────┐ ┌────────────────┐
     │ modules/image  │ │ modules/audio │ │ modules/video  │
     │ ViT + ELA      │ │ Pitch + MFCC  │ │ ViT + SSIM     │
     │ EXIF metadata  │ │ Spectral feat │ │ ffmpeg + ffprobe│
     └───────┬────────┘ └──────┬────────┘ └───────┬────────┘
             │                 │                   │
             └─────────────────┼───────────────────┘
                               │ scored 0–100
                               ▼
                  ┌────────────────────────┐
                  │    fusion/engine.py    │  ← Decision Layer
                  │  0.35×img + 0.25×aud  │  (pure maths, no LLM)
                  │  + 0.25×vid + 0.15×url│
                  │  → verdict + confidence│
                  └────────────┬───────────┘
                               │ numeric verdict
                               ▼
                  ┌────────────────────────┐
                  │      llm/phi3.py       │  ← Explanation Layer
                  │  Phi-3 Mini (Ollama)   │  (words only, no decision)
                  │  Generates narrative   │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  reports/generator.py  │
                  │  PDF Forensic Dossier  │
                  └────────────────────────┘
```

---

## Module Layer

| Module | Primary Signal | Secondary Signal |
|---|---|---|
| `image.py` | ViT AI-image-detector (transformer) | ELA compression artifacts |
| `audio.py` | Pitch monotonicity | MFCC delta smoothness + spectral flatness |
| `video.py` | Per-frame ViT score | SSIM temporal consistency + ffprobe metadata |
| `url.py` | Shannon entropy | Homograph + DGA + shortener detection |
| `threats.py` | MIME ≠ extension | High entropy (packed malware signature) |
| `metadata.py` | EXIF software tag | ffprobe format tags + creation time |

---

## Fusion Weights (Justified)

```python
final_score = (
    0.35 * image_score   # Highest: direct pixel-level AI detection
  + 0.25 * audio_score   # Strong: spectral voice synthesis fingerprint
  + 0.25 * video_score   # Strong: multi-frame + temporal consistency
  + 0.15 * url_score     # Context: supporting evidence only
)
```

---

## Smart Weight Redistribution (when tools unavailable)

```
ffmpeg installed + real metadata → standard 0.6/0.2/0.2 weights
only audio available             → 0.75 visual / 0.25 audio
only metadata (≥20 pts)          → 0.75 visual / 0.25 meta
neither available                → visual takes 1.0 full weight
```

---

## Memory Optimizations (8GB RAM)

| Technique | Where | Effect |
|---|---|---|
| Lazy model loading | `image.py` | ViT loaded once, cached globally |
| Explicit `gc.collect()` | `image.py` | Frees RAM after each image |
| `sr=None` in librosa | `audio.py` | No resampling CPU overhead |
| OpenCV frame extraction | `video.py` | No full video RAM load |
| `num_predict=400` | `phi3.py` | Caps LLM response tokens |
| `temperature=0.1` | `phi3.py` | Fast, deterministic output |
