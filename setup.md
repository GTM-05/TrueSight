# TrueSight System Setup & Prerequisites

This document covers installation for all platforms and explains the dual-app LLM architecture.

---

## 🛠️ 1. Core Prerequisites

| Dependency | Purpose | Required For |
|---|---|---|
| Python 3.10+ | Core runtime | Both modes |
| FFmpeg | Video frame/audio extraction | Both modes |
| Ollama | Local LLM runtime | Both modes |
| `phi` model | Report narration | `app.py` (Standard) |
| `phi3:mini` model | Full evidence reasoning + report | `app-ai.py` (AI-Enhanced) |

---

## 🧠 2. Two Modes, Two LLMs

### Standard Mode (`app.py`) — uses `phi` (Phi-2, ~1.6GB)
The LLM is called **once**, after all analysis is complete, to narrate the final 3-stage PDF report. All actual scoring is done by heuristic algorithms.

### AI-Enhanced Mode (`app-ai.py`) — uses `phi3:mini` (~2.3GB)
The LLM is called **at the fusion stage** — it reads all raw forensic evidence (ELA metrics, EXIF data, acoustic features, URL features) and reasons over them to produce a structured JSON verdict:
```json
{
  "threat_score": 75,
  "ai_generated_score": 20,
  "manipulation_score": 60,
  "final_score": 65,
  "confidence": "High",
  "key_findings": ["Missing EXIF", "High ELA variance"],
  "verdict": "High"
}
```
This verdict then drives both the UI metric breakdown and the 3-stage narrative report.

**Why offline LLMs?** Cyber forensics requires strict data privacy. Sending potentially malicious payloads to cloud APIs is a security violation. Ollama runs everything 100% locally.

---

## 🚀 3. Installation — All Platforms

### Step 1: Clone the Repository
```bash
git clone https://github.com/GTM-05/TrueSight.git
cd TrueSight
```

### Step 2: Install FFmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg -y
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```powershell
choco install ffmpeg
# OR download from https://ffmpeg.org/download.html and add to PATH
```

### Step 3: Install Ollama

**Linux / macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download the installer from [ollama.com](https://ollama.com) and run it.

### Step 4: Pull the LLM Model(s)

For **Standard mode** only:
```bash
ollama pull phi
```

For **AI-Enhanced mode** (or both):
```bash
ollama pull phi3:mini
```

### Step 5: Create Python Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 6: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 7: Start Ollama (if not already running)
```bash
ollama serve
```
*Leave this running in a background terminal.*

---

## ▶️ 4. Running TrueSight

**Standard Mode** (LLM for report narration only):
```bash
streamlit run app.py
```
→ Opens at `http://localhost:8501`

**AI-Enhanced Mode** (LLM reasons over all evidence):
```bash
streamlit run app-ai.py
```
→ Opens at `http://localhost:8501`

> Both apps share the same heuristic modules and PDF generator. Only the fusion + LLM layer differs.
