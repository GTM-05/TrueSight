# 🕵️ TrueSight: Project Overview & Deep Dive

TrueSight is a professional cyber forensics tool designed to detect digital forgery, AI-generated "deepfakes," and malicious payloads hidden in media.

This document provides a **layman's explanation** of how the system works, which models we use, and how the code is structured.

---

## 🌟 1. The Core Concept: "Eyes" vs. "Brain"

To understand TrueSight, imagine it as a detective with two parts:

1.  **The Eyes (Forensic Modules)**: These are specialized Python scripts that look for technical glitches—pixels that don't match, audio frequencies that are too smooth, or web links that look suspicious. These use **Machine Learning (ML)** and **Math**.
2.  **The Brain (LLM/Ollama)**: Once the "Eyes" find clues, they send them to the "Brain." We use **Ollama** (`phi` or `phi3:mini`) as the investigator that reads all the clues and writes a human-readable report explaining the risk.

---

## 🚀 2. Two Modes of Operation

We built TrueSight with two distinct "personalities":

| Mode | Entry File | Ideal For... | Power Level |
|---|---|---|---|
| **Standard** | `app.py` | Quick checks for edited photos or suspicious links. | **Lightweight** (uses basic math) |
| **AI-Enhanced** | `app-ai.py` | Professional deepfake detection (Midjourney, AI-voice, etc). | **Professional** (uses deep learning) |

---

## 🧠 3. The "AI Models" Simplified

We use different types of AI because a single model can't do everything:

-   **Vision Transformer (ViT)**: This is our **Image AI**. It doesn't look for "edits"; it looks for "texture." It knows what a Midjourney image "looks" like at a mathematical level.
-   **Signal Processing (Librosa)**: This is our **Audio AI**. It knows that human voices have "noise" and "jitter," while AI voices are often "too perfect" or "monotone."
-   **Ollama (phi3:mini)**: This is our **Reasoning AI**. It doesn't look at the image; it reads the *data* about the image and decides if the combined evidence is enough to call it a "High Risk."

### 📂 Quick Reference: Model & Modality Map

| Category | Model / Technology | Where it "Lives" |
|---|---|---|
| **🧠 Language** | `phi` and `phi3:mini` | **Ollama** (Shows in `ollama list`) |
| **👁️ Vision** | `ViT` (Vision Transformer) | **Python Libraries** (HuggingFace `transformers` & `torch`) |
| **👂 Audio** | `Acoustic Signal Models` | **Python Libraries** (`librosa`) |
| **📁 Files/Links** | `Heuristic Binary Engines` | **Heuristics** (Custom code in `TrueSight/modules`) |

---

## 🛠️ 5. Technical Implementation & Coding Patterns

For the developers and technical investigators, here is how the code is actually built:

### 🐍 The Python Core
-   **Framework**: We use **Streamlit** for the frontend because it allows for rapid development of data-heavy interfaces without the overhead of JavaScript frameworks.
-   **State Management**: We use `st.session_state` to keep your analysis results persistent as you move between tabs (Video vs. Image vs. URL).
-   **Concurrency**: While the UI is single-threaded, the heavy ML processing happens in block-level executions to keep the interface responsive.

### 🖼️ Image & Video (Computer Vision)
-   **Library**: `OpenCV` (cv2) and `Pillow` (PIL).
-   **ELA Logic**: We re-save the image at 90% quality, then use `np.abs` to calculate the pixel-by-pixel difference. High "noise" in the difference map suggests the area was edited.
-   **ViT Detection**: We use the `transformers` pipeline to load a **Vision Transformer**. This model maps the image into a high-dimensional space to find "synthetic artifacts" that are invisible to the naked eye.
-   **Temporal Analysis (SSIM)**: In video, we compare frame $N$ with frame $N+1$ using the **Structural Similarity Index**. If the SSIM "std dev" is too high, it means the video has unnatural frame transitions—a common deepfake tell.

### 🔊 Audio (Digital Signal Processing)
-   **Library**: `Librosa`.
-   **Feature Engineering**: We don't just "listen." We extract:
    -   **MFCCs**: Captures the "texture" of the voice.
    -   **Pitch (PyIN)**: Checks for the robotic "flatness" of AI voices.
    -   **Spectral Flatness**: Measures how "noisy" vs. "tonal" the sound is.
-   **Anomaly Scoring**: We compare these features against thresholds derived from the WaveFake dataset.

### 🧠 LLM Orchestration
-   **Temperature Control**: We set `temperature: 0.1` for the LLM. 
    -   *Why?* We want the AI to be **deterministic**. It should give the same forensic verdict for the same data every time, not get "creative" with the facts.
-   **Structured Output**: In `app-ai.py`, we force the LLM to think in **JSON**. This allows the Python code to pull out specific numbers (like "85% Threat Score") and show them in those nice big UI metrics.

---

## 🏗️ 6. Full Project Structure

```text
TrueSight/
├── app.py                  # Standard UI (Heuristic Focused)
├── app-ai.py               # AI-Enhanced UI (ML Focused)
├── project.md              # This Master Guide
├── setup.md                # Installation Manual
├── arc.md                  # System Architecture Diagrams
├── requirements.txt        # Python Dependencies
├── modules/
│   ├── image_ai.py         # ViT + ELA logic
│   ├── audio_ai.py         # MFCC + Pitch logic
│   ├── video_ai.py         # Frame sampling + SSIM logic
│   ├── url_ai.py           # Phishing heuristics + TLD parsing
│   ├── threats.py          # Malware & Steganography scanner
│   └── metadata.py         # EXIF & Metadata parser
├── fusion/
│   ├── engine.py           # Simple scoring (Standard)
│   └── engine_ai.py        # Evidence aggregator (AI-Enhanced)
├── llm/
│   ├── phi2.py             # phi-2 prompt engineering
│   └── phi2_ai.py          # phi-3 mini structured reasoning
└── reports/
    └── generator.py        # ReportLab PDF building logic
```

---

## 🚀 7. The Forensic Pipeline (How it works)

1.  **Ingestion**: Streamlit handles the file upload/URL input.
2.  **Pre-flight**: `threats.py` scans for embedded malware or steganography.
3.  **Extraction**: Heuristic and ML modules extract numerical "features" (ELA, MFCC, AI-prob).
4.  **Reasoning**: `engine_ai.py` bundles these features into a JSON "evidence bag" and sends it to `phi3:mini`.
5.  **Verdict**: The LLM parses the evidence, assigns category scores, and writes the narrative.
6.  **Documentation**: `generator.py` compiles everything into a "Clean & Neat" PDF report.

---

## 🛡️ 8. Why This Design Matters
Cyber forensics requires **Privacy** and **Reproducibility**. 
By staying **Offline-First**, TrueSight ensures that sensitive evidence never leaves your machine. By using **Deterministic AI**, it ensures that your investigation is scientific and repeatable.
