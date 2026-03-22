# TrueSight Setup Guide

Follow these steps to deploy a local, privacy-hardened forensics environment.

## 1. Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 22.04+ Recommended)
- **RAM**: 8GB Minimum (4GB works in "Lite AI Mode")
- **Disk**: 5GB for models and dependencies.

### System Dependencies
Install FFmpeg and Python headers:
```bash
sudo apt update
sudo apt install ffmpeg python3-dev python3-pip -y
```

---

## 2. AI Brain Installation (Ollama)

TrueSight uses a local LLM for narrative analysis.
1.  **Install Ollama**:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
2.  **Pull the Forensic Models**:
    TrueSight defaults to `qwen2:0.5b` for 8GB RAM systems.
    ```bash
    ollama pull qwen2:0.5b
    ```

---

## 3. Python Environment Setup

1.  **Create a Virtual Environment**:
    ```bash
    cd TrueSight
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## 4. Running the Application

### Method A: Single Command (Standard)
```bash
# Start Ollama engine in background
ollama serve &

# Launch Streamlit UI
streamlit run app.py
```

### Method B: Lite Mode (Low Resource)
If you have < 6GB RAM, launch with:
```bash
# In the UI sidebar, toggle 'Low Resource Mode'
# This disables SSIM and reduces ViT frame sampling.
```

---

## 5. Troubleshooting

- **Ollama Connection Error**: Ensure `ollama serve` is running. Check with `curl http://localhost:11434`.
- **CV2/Import Errors**: Ensure you are in the `venv`. Re-run `pip install -r requirements.txt`.
- **FFmpeg Not Found**: Run `ffmpeg -version` to verify system installation.

---

## 6. Project Credentials
- **License**: Private/Internal Use
- **Author**: GTM-05
