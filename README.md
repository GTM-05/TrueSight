# 🕵️ TrueSight: Multimodal Cyber Forensics

## 📌 Project Summary
TrueSight is an AI-powered, lightweight desktop forensic application engineered to analyze digital media and web links for malicious manipulation, synthetic AI generation, and embedded cyber threats. 

Built specifically for the rigors of modern cyber-investigations, TrueSight performs **offline heuristic modeling** (like Error Level Analysis, acoustic fingerprinting, and binary malware scanning) combined with an intelligent **3-Stage LLM Fusion Engine** to deliver a comprehensive, investigator-ready PDF report.

---

## 🏛️ Clean System Architecture

The tool is decoupled into distinct processing pipelines:

1. **Frontend Dashboard (`app.py`)**: A Streamlit interface where investigators upload potentially compromised media or URLs.
2. **Heuristic Engines (`modules/`)**:
   - `threats.py`: Performs rigid binary scans on all uploads to trap steganography payloads, executable magic bytes, and polyglot web-shells before rendering.
   - `image.py & metadata.py`: Executes Error Level Analysis (ELA) to find Photoshop splicing, and flags zero-EXIF + native resolutions to catch AI-generated images (e.g., Midjourney/DALL-E).
   - `video.py`: Extracts exact 10-second segments to map temporal anomalies and explicitly penalizes raw, silent AI outputs (like OpenAI Sora constraints).
   - `audio.py`: Evaluates Mel-Frequency Cepstral Coefficients (MFCCs) and spectral flatness to instantly catch cloned, synthetic, or text-to-speech engine outputs.
   - `url.py`: Assesses domain age, entropy, and lexical strings to compute phishing risk scores out of 100%.
3. **Fusion & Reasoning (`fusion/` & `llm/`)**: All isolated metric scores are pushed to `fusion/engine.py` to synthesize a global Risk Verdict. That fused data array is piped into Microsoft's local **Phi-2 AI Model** (`llm/phi2.py`) to generate a human-readable **3-Stage Report** (Threat Assessment, AI-Generation Assessment, and Deepfake/Manipulation Assessment).
4. **Reporting (`reports/`)**: A physical PDF dossier is built with `ReportLab` allowing the user to export their findings securely.

---

## 🎯 Where Are We Using This?
TrueSight is positioned for:
* **Cybersecurity Operations Centers (SOC):** Quickly triaging incoming media files for embedded `.php` or `.exe` payloads disguised as JPEGs.
* **Journalism & Fact-Checking:** Verifying whether a viral news video was naturally captured on a smartphone or synthetically baked by an AI API.
* **Academic/College Demonstration:** Perfect for live forensic showcase testing, featuring distinct thresholds to separate untouched human media from algorithmic fakes.

---

## 🚀 How to Run & Test (Project Showcase Guide)

This repository is completely self-contained. 

### 1. Start the System
If you are running this natively for the first time, ensure dependencies are installed via `pip install -r requirements.txt`. To fire up the investigator dashboard:
```bash
streamlit run app.py
```
*(Optional: Make sure Ollama is installed and running `ollama run phi` in the background if you want the textual AI summaries to populate!)*

### 2. How to Perform the Live Showcase test
Included in the repository is a highly curated folder named `Final_Project_Showcase_Samples/` containing guaranteed **100% Authentic Human** and **100% Synthetic AI** files. 

1. **Open the Dashboard:** Navigate to `http://localhost:8501`.
2. **Run an Audio Test:** 
   - Upload `Showcase_Audio_Human.wav` -> **Expected Result:** ~20% (Low Risk / Clean).
   - Upload `Showcase_Audio_AI.wav` -> **Expected Result:** ~75% (High Risk / Synthetic Detected).
3. **Run an Image Test:**
   - Upload `Showcase_Image_Human.jpg` -> Watch the metadata engine clear all ELA anomalies.
   - Upload `Showcase_Image_AI.jpg` -> Watch the engine flag the 1024x1024 missing-EXIF heuristic.
4. **Export Findings:** Click the **Generate Final Forensic Report** button at the bottom of the screen to dynamically route all findings to the Phi-2 AI model for your concluding 3-stage PDF!
# TrueSight
