# 🕵️ TrueSight: Multimodal Cyber Forensics

## 📌 Project Summary
TrueSight is an AI-powered, lightweight desktop forensic application engineered to analyze digital media and web links for malicious manipulation, synthetic AI generation, and embedded cyber threats.

Built for modern cyber-investigations, TrueSight performs **offline heuristic modeling** (Error Level Analysis, acoustic fingerprinting, binary malware scanning) with a local LLM for forensic reporting. It ships in **two modes**:

| Mode | Entry Point | LLM Role | Model |
|---|---|---|---|
| **Standard** | `app.py` | Narrates the final report only | `phi` (Phi-2) |
| **AI-Enhanced** | `app-ai.py` | Reasons over all evidence at fusion stage, produces per-category scores | `phi3:mini` |

---

## 🏛️ System Architecture

### Standard Mode (`app.py`)
```
Upload/URL → Threat Scan → Heuristic Engines → Score Averaging (fusion/engine.py)
          → phi narrates 3-stage report → PDF Export
```

### AI-Enhanced Mode (`app-ai.py`)
```
Upload/URL → Threat Scan → Heuristic Engines → Full Evidence Dict
          → phi3:mini reasons over all evidence (fusion/engine_ai.py)
          → Structured JSON Verdict { threat_score, ai_generated_score, manipulation_score }
          → phi3:mini narrates 3-stage report → PDF Export
```

### Heuristic Engines (`modules/`)
- **`threats.py`** — Binary scans: magic bytes, polyglot shells, steganography payloads
- **`image.py` & `metadata.py`** — Error Level Analysis (ELA), EXIF inspection, blur detection
- **`video.py`** — Frame-level temporal analysis, optical flow, silent track detection
- **`audio.py`** — MFCC, spectral flatness, SNR — catches TTS / cloned voice
- **`url.py`** — Domain entropy, lexical phishing heuristics

### Module Map
```
llm/
  phi2.py          ← Standard: generate_ai_explanation(modality_data)
  phi2_ai.py       ← AI-Enhanced: llm_reason_verdict() + generate_ai_explanation(verdict, evidence)

fusion/
  engine.py        ← Standard: simple score averaging
  engine_ai.py     ← AI-Enhanced: LLM-powered structured verdict
```

---

## 🎯 Use Cases
- **SOC / Incident Response** — Triage media files for embedded payloads disguised as JPEGs or MP4s
- **Journalism / Fact-Checking** — Detect whether viral footage is authentic or AI-synthesized
- **Academic Showcase** — Live demonstration of multi-modal forensic heuristics

---

## 🚀 Quick Start

> Full installation steps → [setup.md](setup.md) | Architecture details → [arc.md](arc.md)

```bash
git clone https://github.com/GTM-05/TrueSight.git
cd TrueSight
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Standard mode (phi model)
ollama pull phi
streamlit run app.py

# AI-Enhanced mode (phi3:mini — ~2.3GB download)
ollama pull phi3:mini
streamlit run app-ai.py
```

---

## 📊 Expected Test Results

Upload the sample files from `test_samples/` to validate each engine:

| File Type | Human Sample | AI/Fake Sample |
|---|---|---|
| Audio | ~20% (Low Risk) | ~75% (High Risk) |
| Image | Low ELA variance, rich EXIF | Zero EXIF, uniform ELA, 1024×1024 |
| URL | — | High entropy, suspicious TLDs |
