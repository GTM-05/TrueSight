# TrueSight AI — Multimodal Cyber Forensics (v3.0)

> **Local. Private. Heuristic-first.** Scoring and fusion are deterministic; the LLM only narrates from structured facts.

TrueSight detects AI-generated, deepfaked, or manipulated **image, audio, video, and URL** content using computer vision, signal processing, and optional **Ollama / Qwen** text for human-readable reports. All heavy scoring runs offline on your machine.

---

## How it works

```
Upload → Per-modality analyzers → Fusion engine (math) → LLM narrative (optional) → PDF / UI
```

1. **Image** — ViT-style detector, ELA, spectral slope, chroma/noise/DCT/copy-move (some checks skipped or retuned for **video frames** when `source="video"`).
2. **Video** — Sampled frames analyzed as video-sourced images; **face ROI** adjacent-frame SSIM (dual-threshold morphing), optical-flow **face warp**, rPPG liveness, lip–audio sync, metadata; **`morphing_score`** combines spatial signals, metadata scale, and **audio phase spike** density (not the same as aggregate video risk).
3. **Audio** — Pitch, MFCC, HNR, spectral flatness, **phase discontinuities** (splice spikes), silence/TTS heuristics.
4. **URL** — Homograph, entropy, shorteners, TLS, phishing keywords (RFC1918 / localhost handling).
5. **Fusion** (`fusion/engine.py`) — **Consensus-boosted** (CDC) strong-anchor fusion; large scores anchored when 4+ sectors fire. **Liveness-gating** prevents metabolic spoofing by skipping reduction when structural forensics (ELA, AI Gen) are active.
6. **LLM** (`llm/llm.py`) — Receives a **JSON forensic brief** to ensure narrative consistency with numeric scores.

**UI:** Per-frame indicator floods are grouped by tag via **`display_indicators()`** in `app.py` (e.g. many `[ELA]` lines → one summary).

---

## Quick start

```bash
git clone https://github.com/GTM-05/TrueSight.git && cd TrueSight

sudo apt install ffmpeg -y

curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:3b    # or: ollama pull qwen2:0.5b  (see config.py)

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

**Regression / smoke:** `python verify_accuracy.py` (optional `--strict-benchmarks` for labeled clips in `samples/`).

---

## System requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB+ |
| CPU | 4-core | 6-core+ |
| Storage | ~5 GB | ~10 GB+ (ViT + optional LLM) |
| Python | 3.10+ | 3.12+ |
| OS | Linux (Ubuntu 22.04+) | Ubuntu 24.04 |

---

## Repository layout

```
TrueSight/
├── app.py                 # Streamlit UI, fusion + LLM orchestration
├── config.py              # ForensicConfig (CFG) — thresholds, fusion, LLM defaults
├── verify_accuracy.py     # Video + fusion smoke / benchmark harness
├── modules/
│   ├── image.py           # Image + video-frame paths (ELA gates differ by source)
│   ├── video.py           # Sampling, liveness, morphing components, audio extract
│   ├── audio.py           # Detectors + sub_scores (e.g. phase spike_count)
│   ├── url.py
│   ├── metadata.py
│   └── threats.py
├── fusion/engine.py       # compute_final_score, compute_morphing_score, verdict adapter
├── llm/
│   ├── llm.py             # Ollama narrative from structured JSON brief
│   └── report_generator.py # Alternate PDF path with Ollama section (optional)
├── reports/generator.py   # UI “dossier” PDF generator (ReportLab)
└── docs in repo root: ARCHITECTURE.md, ALGORITHM.md, PROJECT_SPEC.md, SETUP.md, prompt.md
```

---

## Documentation

| File | Purpose |
|------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Modules, fusion lifecycle, patterns |
| [ALGORITHM.md](ALGORITHM.md) | Pipeline stages, floors, video/morphing signals |
| [PROJECT_SPEC.md](PROJECT_SPEC.md) | Config highlights and verdict bands |
| [SETUP.md](SETUP.md) | Install, venv, Ollama, troubleshooting |
| [prompt.md](prompt.md) | Maintainer DNA / onboarding |
| [workflows/optimized_forensics.md](workflows/optimized_forensics.md) | Operator-style workflow notes |
