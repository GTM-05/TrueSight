# TrueSight — Setup Guide

## Prerequisites (All Platforms)

Before you start, install these **in order**:

### 1. Python 3.10+

| OS | Command |
|---|---|
| Ubuntu/Linux | `sudo apt install python3 python3-pip python3-venv -y` |
| macOS | `brew install python` |
| Windows | Download from [python.org](https://www.python.org/downloads/) — check **Add to PATH** |

Verify: `python3 --version`

---

### 2. FFmpeg (Required for audio + video metadata)

> ⚠️ Without ffmpeg, audio analysis and video metadata scanning are disabled — accuracy drops significantly.

| OS | Command |
|---|---|
| Ubuntu/Linux | `sudo apt install ffmpeg -y` |
| macOS | `brew install ffmpeg` |
| Windows | `choco install ffmpeg` OR download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH |

Verify: `ffmpeg -version` and `ffprobe -version`

---

### 3. Ollama (Local AI Brain)

| OS | Command |
|---|---|
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| macOS | Download [Ollama.app](https://ollama.com/download) |
| Windows | Download [OllamaSetup.exe](https://ollama.com/download) |

After install, pull the model (required for AI reports):
```bash
ollama pull qwen2:0.5b   # Lite (350MB) - Recommended for 8GB RAM
```
Verify: `ollama list` — you should see `qwen2:0.5b`

> [!TIP]
> If `ollama pull` fails, ensure the Ollama server is running (see Step 1 of "Running the App").

---

### Summary Checklist

| Tool | Check Command | Required |
|---|---|---|
| Python 3.10+ | `python3 --version` | ✅ |
| pip | `pip --version` | ✅ |
| ffmpeg | `ffmpeg -version` | ✅ (accuracy) |
| ffprobe | `ffprobe -version` | ✅ (accuracy) |
| Ollama | `ollama list` | ✅ (reports) |
| qwen2:0.5b | `ollama list` shows qwen2 | ✅ (reports) |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/GTM-05/TrueSight.git
cd TrueSight

# Create virtual environment
python3 -m venv venv

# Activate (see OS-specific below)

# Install Python packages
pip install -r requirements.txt
```

### Activate Virtual Environment

| OS | Command |
|---|---|
| Linux / macOS | `source venv/bin/activate` |
| Windows (CMD) | `venv\Scripts\activate.bat` |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |

---

## Running the App

### Step 1 — Start Ollama (if not running)

| OS | Command |
|---|---|
| Linux / macOS | `ollama serve &` |
| Windows | Ollama runs automatically in the system tray |

### Step 2 — Launch TrueSight

**Method A: Standard (Requires Virtual Env Activation)**
```bash
streamlit run app.py
```

**Method B: Full Command (Most Reliable — works without manual activation)**
| OS | Command |
|---|---|
| Linux / macOS | `./venv/bin/python3 -m streamlit run app.py` |
| Windows | `.\venv\Scripts\python.exe -m streamlit run app.py` |

Then open: **http://localhost:8501**

---

## Kill / Stop / Reset

### Kill the Streamlit App

| OS | Command |
|---|---|
| Linux / macOS | `fuser -k 8501/tcp` |
| macOS (alternative) | `lsof -ti:8501 \| xargs kill -9` |
| Windows (CMD) | `netstat -ano \| findstr :8501` → note PID → `taskkill /PID <pid> /F` |
| Windows (PowerShell) | `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess -Force` |

### Kill Ollama

| OS | Command |
|---|---|
| Linux | `pkill ollama` |
| macOS | `pkill ollama` |
| Windows | Right-click Ollama in system tray → Quit |

### Delete Temp Files (cleanup)

| OS | Command |
|---|---|
| Linux / macOS | `rm -f /tmp/tmp*.jpg /tmp/tmp*.mp4 /tmp/tmp*.wav` |
| Windows (PowerShell) | `Remove-Item "$env:TEMP\tmp*" -Force -ErrorAction SilentlyContinue` |

### Full Reset (wipe venv + restart fresh)

| OS | Command |
|---|---|
| Linux / macOS | `deactivate; rm -rf venv; python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` |
| Windows (PowerShell) | `deactivate; Remove-Item venv -Recurse -Force; python -m venv venv; venv\Scripts\Activate.ps1; pip install -r requirements.txt` |

---

## Troubleshooting

### "streamlit: command not found"
If you get this error even after installing, it means `streamlit` is not in your global system path.
**Solution:** Always use the "Full Command" (Method B as shown above):
`./venv/bin/python3 -m streamlit run app.py` (Linux/macOS)
`.\venv\Scripts\python.exe -m streamlit run app.py` (Windows)

### "python: command not found"
On many Linux systems, use `python3` instead of `python`.

### "HTTP 500 Error" or "Ollama Connection Failed"
This usually happens if the Ollama server isn't running or hasn't finished loading the model.
1. Check if it's running: `curl http://localhost:11434/api/tags`
2. If it returns an error, start it: `ollama serve &`
3. Ensure you have the model: `ollama pull qwen2:0.5b`

---

## Performance Tips (8GB RAM)

| Setting | Recommendation |
|---|---|
| Enable **Low Resource Mode** sidebar checkbox | Reduces to 1 frame, skips SSIM |
| Close other apps before video analysis | Frees RAM for ViT model |
| Use image tab before video | Preloads ViT model into RAM cache |

### RAM Usage by Component

| Module | Approx. RAM |
|---|---|
| ViT Image Detector | 200–300 MB |
| Audio Analysis (librosa) | 80–100 MB |
| Video Frame Analysis | 300–400 MB |
| AI Analyst (Qwen2) | 300–500 MB |
| **Total** | **~1.0–1.5 GB** |
