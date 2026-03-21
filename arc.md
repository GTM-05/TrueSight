# TrueSight: System Architecture and Setup Guide

## 🏗️ System Architecture Diagram

### Text-Based Architecture (ASCII)

```text
[User Media Upload / URL Input] 
           │
           ▼
[Binary Malware Scanner (threats.py)] ──(If Malware Detected)──► [CRITICAL THREAT ALERT]
           │
     (If File Safe)
           ▼
[Heuristic Data Router] ──► (Routes based on file type)
           │
           ├─► Video (video.py) ----.
           ├─► Image (image.py) ----+
           ├─► Audio (audio.py) ----+
           ├─► Meta (metadata.py) --+
           └─► URL (url.py) --------'
                                    │
                                    ▼
                      [Risk Scoring Fusion Engine] (fusion/engine.py)
                                    │
                                    ▼
                      [Local LLM (Phi-2 via Ollama)] (llm/phi2.py)
                         (Analyzes specific metrics)
                                    │
                                    ▼
       [3-Stage Forensic Dossier (Threat, AI Origin, Manipulation Info)]
                                    │
                                    ▼
                 [Final Dashboard Verdict & PDF Export Generator]
```

### Flowchart Architecture (Markdown/Mermaid)
This flowchart outlines the exact pipeline of how a file travels from the user upload through our advanced, multi-modal forensic engines, to the AI text synthesizer, and finally into the final PDF report.

```mermaid
graph TD
    %% User Input Layer
    User[Investigator] -->|Uploads Media / Enter URL| WebUI(Streamlit UI: app.py)

    %% Threat Gateway Layer
    WebUI -->|Binary Validation| ThreatModule[modules/threats.py]
    ThreatModule -- Detects Payload --> Alert[CRITICAL THREAT FLAG]
    ThreatModule -- Media Verified Safe --> Router{Heuristic Data Router}

    %% Heuristic Processor Layer
    Router -->|Video (.mp4, .avi)| ModVideo[modules/video.py]
    Router -->|Image (.jpg, .png)| ModImage[modules/image.py]
    Router -->|Image Metadata| ModMeta[modules/metadata.py]
    Router -->|Audio (.wav, .mp3)| ModAudio[modules/audio.py]
    Router -->|Hyperlinks| ModURL[modules/url.py]

    %% Deep Processing Sub-routines
    ModVideo -.->|Internal Extraction| ModAudio
    ModVideo -.->|Frame Slicing| ModImage
    
    %% Aggregation Layer
    ModVideo --> FusionEngine((fusion/engine.py))
    ModImage --> FusionEngine
    ModMeta --> FusionEngine
    ModAudio --> FusionEngine
    ModURL --> FusionEngine

    %% AI Pipeline Layer
    FusionEngine -- Aggregated Risk Scores --> LLM[llm/phi2.py : Local AI Model]
    
    %% Output Generation Layer
    LLM -- 3-Stage Reasoning Output --> ReportGen[reports/generator.py]
    ReportGen --> Output[[Final PDF Forensic Dossier]]
    Alert --> Output
```

---

## 🛠️ Pre-requisites
Before assembling the TrueSight project on your machine, ensure you have the following system requirements:
* **Python 3.10+**: Recommended for strict library dependencies (`librosa`, `moviepy`).
* **Pip / Virtualenv**: Required for environment isolation.
* **Ollama (Optional but Recommended)**: The project operates fully offline, requiring [Ollama](https://ollama.com/) running locally to generate the 3-Stage AI textual reasoning summaries using the `phi` (Phi-2) modal.
* **FFmpeg**: Required internally by the `moviepy` library to extract frames and audio segments from `.mp4` payloads. 
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: Install via Chocolatey or download binaries.

---

## 🚀 Setup & Installation Steps

Execute the following terminal commands to deploy the forensics engine directly from a fresh repository clone:

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/TrueSight.git
cd TrueSight
```

**2. Isolate the Python Environment**
It is extremely critical when utilizing deep-learning and acoustic libraries to employ an isolated environment.
```bash
python3 -m venv venv
```

**3. Activate the Environment**
* On **Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```
* On **Windows**:
  ```cmd
  venv\Scripts\activate
  ```

**4. Install Data Dependencies**
This command triggers the installation of the core mathematical heuristics (OpenCV, Librosa, MoviePy, Streamlit).
```bash
pip install -r requirements.txt
```

**5. Pull the Local LLM**
*(Assuming Ollama is natively installed via `curl -fsSL https://ollama.com/install.sh | sh`)*
Pull the highly efficient Phi-2 neural network locally:
```bash
ollama run phi
```
*(You can safely hit `Ctrl+C` to exit the chat terminal once the model finishes downloading).*

**6. Deploy TrueSight**
Boot up the Streamlit dashboard on your local machine port.
```bash
streamlit run app.py
```
