# TrueSight System Setup & Prerequisites

This document outlines the strict installation instructions and explains the architectural decisions behind deploying local LLMs for the TrueSight Cyber Forensics tool.

---

## 🛠️ 1. Core Prerequisites

Before deploying TrueSight on any machine, the following base dependencies must be met:

1. **Python 3.10+**: Requires a modern Python environment to support structural acoustic and computer vision libraries.
2. **FFmpeg**: A background C-library required to slice video files into frames and audio strips.
   - **Linux**: `sudo apt install ffmpeg`
   - **Mac**: `brew install ffmpeg`
   - **Windows**: Download binaries and add to PATH or `choco install ffmpeg`
3. **Ollama**: The local AI runtime environment.

---

## 🧠 2. The Ollama Phi-2 Integration

### "Are we using Ollama Phi-2?"
**Yes, absolutely.** TrueSight connects directly to the **Phi-2** neural network via Ollama for its final reporting phase. 

Whenever you click "Generate Final Forensic Report", the application packages the exact heuristic scores (from the malware, audio, and visual engines) and sends them to your local Ollama daemon. The Phi-2 model then acts as the "AI Detective", writing out the structured **3-Stage Forensic Dossier** explaining *why* a file is AI-generated or malicious. 

### Why Phi-2 specifically?
* **Privacy & Security:** Cyber forensics requires strict data privacy. Sending malicious payloads or sensitive investigation photos to a cloud API (like ChatGPT) is a security violation. Ollama runs Phi-2 100% offline and locally on your hardware.
* **Lightweight ("Lite" Constraint):** Phi-2 is a massively capable 2.7-billion parameter model that easily runs on standard college laptops without requiring expensive cloud GPUs.
* **Graceful Degradation:** If the Ollama server crashes or isn't started by the user, TrueSight is programmed to gracefully degrade—meaning it will still output the flawless mathematical heuristic scores on the dashboard, it simply skips writing the textual AI summary.

---

## 🚀 3. Step-by-Step Installation Guide

**Step 1: Clone the Project**
```bash
git clone https://github.com/your-username/TrueSight.git
cd TrueSight
```

**Step 2: Start the AI Investigator (Ollama)**
Download Ollama from [ollama.com](https://ollama.com). Ensure the background service is running, and pull the Microsoft Phi-2 model into your local machine:
```bash
ollama pull phi
```
*Leave the Ollama service running in the background.*

**Step 3: Setup Python Environment**
It is highly recommended to isolate the dependencies.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 5: Launch TrueSight**
```bash
streamlit run app.py
```
*The dashboard will instantly open in your web browser at `http://localhost:8501`, fully connected to the local Phi-2 model.*
