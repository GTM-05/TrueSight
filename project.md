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

---

## 📂 4. Module-by-Module Breakdown

Here is what each folder in the project does:

### 📁 `modules/` (The Detectors)
-   **`image_ai.py`**: Runs the ViT Model to catch AI-generated images.
-   **`audio_ai.py`**: Analyzes the "vibe" of audio to catch cloned voices.
-   **`video_ai.py`**: Breaks video into frames and checks each one for AI-generation.
-   **`url_ai.py`**: Checks if a web link is trying to trick you (e.g., "paypa1.com").
-   **`metadata.py`**: Inspects "hidden data" (EXIF) to see which camera took the photo.
-   **`threats.py`**: A digital "baggage scanner" that looks for malware hidden inside files.

### 📁 `fusion/` (The Decision Maker)
-   **`engine_ai.py`**: Collects all the results from the modules above and prepares them for the LLM.

### 📁 `llm/` (The Narrator)
-   **`phi2_ai.py`**: The bridge to Ollama. It tells the AI exactly how to write the forensic report.

### 📁 `reports/` (The Final Product)
-   **`generator.py`**: Takes the AI's words and the module's scores and builds a professional **PDF Forensic Dossier**.

---

## 🛠️ 5. The Forensic Pipeline (How a file is analyzed)

1.  **Upload**: You drop a file into the Streamlit UI.
2.  **Threat Scan**: Before analyzing for "AI," we check if the file is a virus (`threats.py`).
3.  **Deeper Analysis**: The file goes to the Image, Audio, or Video module.
4.  **Evidence Collection**: Scores and "reasons" (e.g., "Missing EXIF data") are gathered.
5.  **AI Reasoning**: The data is sent to Ollama. The AI "thinks" and provides a verdict.
6.  **Final Report**: You get a PDF with all the technical details and a high-level summary.

---

## 🛡️ 6. Why it's Secure
TrueSight is **100% Offline**. 
- Your images are never sent to a cloud.
- The AI runs on your own computer via Ollama.
- This is critical for forensics because you don't want to leak sensitive investigation data to a third-party company.
