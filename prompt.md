# TrueSight: Project DNA & Onboarding Prompt

You are tasked with maintaining or extending **TrueSight**, a privacy-first, offline, multi-modal cyber forensic investigator.

## 🧠 Core Identity & Philosophy
TrueSight is built on the principle of **"Heuristic-First Forensics."** Unlike typical black-box AI detectors, TrueSight prioritizes mathematical "dead giveaways" (artifact signatures) across Image, Audio, Video, and Metadata. It uses lightweight, local AI (Transformers/Ollama) to *support* findings rather than act as the sole source of truth.

## 🛡️ The "Strong Accurate Algorithm"
The system utilizes a multi-stage cascading filter to ensure high precision:

1.  **Fast Heuristics**: Laplacian variance, fixed AI resolutions, and basic metadata tags.
2.  **Advanced Signal Processing**:
    - **Radial Spectral Slope**: Detects 1/f² power law deviations (AI synthesis signatures).
    - **Chromatic Alignment**: Detects suspiciously perfect R/B channel alignment.
    - **Noise Floor Analysis**: Identifies sterile digital noise vs. natural sensor grain.
3.  **Biological Vitality (Primary Truth)**:
    - **CHROM-rPPG**: Heartbeat detection via chrominance-based pulse extraction (lighting-robust).
    - **Iris Jitter**: Detection of static/frozen gaze (lack of microsaccades).
    - **Blink Calibration**: Frequency and timing of blinks.
4.  **Max-Biased Fusion Engine**: Anomalies are NOT averaged out. A single high-confidence threat (e.g., Grids or Pulse Anomaly) will drive the final risk score high, regardless of other clean signals.
5.  **The Safety Floor**: Samples lacking strong forensic "anchors" are capped at **19%** (Fixed Low Risk) to eliminate "noisy accumulator" false positives.

## 🛠️ Technology Stack
- **Framework**: Streamlit (Python)
- **Computer Vision**: OpenCV (ROI-Face-Tracking, FFT, ELA, Flow)
- **Audio Processing**: Librosa (Pitch Jitter, Spectral Flatness, RMS Flux)
- **AI Backend**: 
    - Transformers (Offline ViT-based AI Detector)
    - Ollama (Qwen2/Phi-2 for Forensic Narrative generation)
- **Reporting**: ReportLab (PDF)

## 📁 Project Structure
- `modules/image.py`: Low-level artifact detection (ELA, Spectral, Noise).
- `modules/video.py`: ROI-based biological scanning (Liveness, Jitter, Flow).
- `modules/audio.py`: Vocal tract forensics (Monotonicity, Flatness).
- `fusion/engine.py`: Weighted risk logic and Max-Biased aggregation.
- `app.py`: Streamlit frontend and forensic triage orchestration.

## 🔄 Operational Workflow
1.  **Ingestion**: User uploads Multi-modal (Video/Image/Audio) or URL.
2.  **Preprocessing**: FFmpeg extracts audio streams and frame sequences (6fps-30ps depending on resource mode).
3.  **Modular Analysis**: Parallel execution of `image`, `video`, and `metadata` modules.
4.  **Biological Triage**: Face tracking isolates ROI -> CHROM-rPPG verifies heart rate -> Iris Jitter verifies gaze.
5.  **Weighted Fusion**: `fusion/engine.py` applies the max-biased logic and safety floor caps.
6.  **Narrative Synthesis**: Data is passed to `llm/llm.py` (Local Ollama) to generate a human-readable forensic explanation.
7.  **Final Reporting**: Verification metadata is generated as a structured PDF.

## 🎯 Developer Directives
- **Lite over Bloat**: Avoid adding massive deep learning models. Prefer mathematical heuristics.
- **Privacy-First**: No external API calls except for URL scanning (if enabled). Ensure `HF_HUB_OFFLINE=1`.
- **Explainability**: Every scoring increase MUST have a corresponding string reason in the `reasons` list.
