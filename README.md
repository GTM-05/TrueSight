# TrueSight AI — Cyber Forensics Tool

> **Local. Private. AI-Powered.** No data ever leaves your machine.

A multimodal cyber forensics tool that detects AI-generated, deepfaked, or manipulated media using a combination of computer vision, signal processing, and a local LLM (Qwen2 0.5B) for report generation.

---

## How It Works

```
Upload Media → Multi-Stage Analysis → Forensic Fusion → Qwen2 Analyst → PDF Report
```

1.  **AI Image Detection**: ViT transformers + Elite Spectral FFT detects internal "grids" left by GANs/Diffusion.
2.  **Biological Liveness**: Monitors rPPG pulse (skin chrominance) and blink variance to confirm human presence.
3.  **Spatio-Temporal Video**: SSIM and Optical Flow analyze frame-to-frame consistency for deepfake warping.
4.  **Audio Forensics**: Pitch, MFCC, and spectral energy analysis catches synthetic TTS and voice clones.
5.  **Forensic Fusion**: A cascading logic engine that applys "Low-Confidence Capping" (19% safety floor) to prevent false positives.
6.  **Qwen2 (0.5B)**: Local LLM writes the final forensic explanation with zero cloud dependency.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/GTM-05/TrueSight.git && cd TrueSight

# 2. Install system tools
sudo apt install ffmpeg -y

# 3. Install AI brain
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2:0.5b

# 4. Python setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 5. Launch
ollama serve &
streamlit run app.py
```

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB+ |
| CPU | 4-core | 6-core+ |
| Storage | 5 GB | 10 GB |
| Python | 3.10+ | 3.12+ |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |

---

## Project Structure

```
TrueSight/
├── app.py                  ← Main UI (single entry point)
├── modules/
│   ├── image.py            ← ViT + ELA + EXIF
│   ├── audio.py            ← Pitch + MFCC + Spectral
│   ├── video.py            ← Frame AI + SSIM + ffprobe
│   ├── url.py              ← Entropy + Homograph + DGA
│   ├── metadata.py         ← EXIF + video container tags
│   └── threats.py          ← Malware signature scan
├── fusion/
│   └── engine.py           ← Weighted decision (pure maths)
├── llm/
│   └── llm.py              ← Qwen2 explanation layer
├── reports/
│   └── generator.py        ← PDF dossier generator
├── code.md                 ← Full code flow reference
├── arc.md                  ← Architecture diagrams
├── project.md              ← Project reference + scoring tables
└── setup.md                ← Detailed setup guide
```

---

## Documentation

| File | Purpose |
|---|---|
| [`code.md`](code.md) | Explains every file's logic and data flow |
| [`arc.md`](arc.md) | Architecture diagrams and weight justification |
| [`project.md`](project.md) | Scoring thresholds and design decisions |
| [`setup.md`](setup.md) | Step-by-step installation guide |
