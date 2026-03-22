# TrueSight Architecture Decision Record

This document outlines the system design and architectural patterns for TrueSight AI.

## 1. System Philosophy
TrueSight is designed to be **Privacy-First** and **Resource-Efficient**. It runs entirely on local hardware (optimized for 8GB RAM) and uses specialized forensic heuristics to augment small-scale AI models.

## 2. Core Modules

### 2.1 Analysis Layer (`modules/`)
These modules extract raw forensic evidence without using the LLM.
- **`video.py`**: The "High-Traffic" entry point. Orchestrates frame sampling, rPPG liveness, and SSIM consistency.
- **`image.py`**: Handles static analysis using ViT transformers and ELA.
- **`audio.py`**: Performs signal processing (Pitch std, MFCC, Energy) to catch voice clones.
- **`metadata.py`**: Cross-references internal file tags with known generative AI tool signatures.
- **`url.py`**: Analyzes entropy, DGA probability, and homograph similarity for phishing detection.
- **`threats.py`**: A malware scanner that provides an "Immediate-Fail" override for suspicious binaries.

### 2.2 Fusion Intelligence (`fusion/`)
- **`engine.py`**: Implements the weighted decision logic. It takes raw scores from the Analysis Layer and combines them into three categorical metrics: **Threat**, **AI-Generated**, and **Manipulation**.

### 2.3 The Analyst Layer (`llm/`)
- **`llm.py`**: Interfaces with Ollama to run the `qwen2:0.5b` model. This layer acts as the "Narrator," turning mathematical scores into a human-readable forensic dossier.

## 3. Data Flow (Lifecycle of an Analysis)

1.  **Ingestion**: Streamlit UI accepts a file upload.
2.  **Taping**: `video.py` or `image.py` performs initial heuristics (blur, format).
3.  **Cascading Filter**: If suspicious, it triggers the "Strong Accurate Algorithm" (SSIM, FFT Grids, Liveness).
4.  **Fusion**: Results are vectorized and passed to `fusion/engine.py`.
5.  **Narration**: LLM receives a context string (Scoring Table + Evidence list) and generates the "Dossier."
6.  **Archival**: `reports/generator.py` serializes the result into a timestamped PDF.

## 4. Key Architectural Patterns

- **Cascading Precision**: Expensive ML models (ViT) are only run on a subset of "Candidate Frames" to preserve RAM.
- **The 19% Safety Floor**: A design pattern that prevents "False Positive Creep" by capping scores that lack definitive forensic markers.
- **Stateless Analysis**: Every request is isolated. The `tempfile` module is used for frame extraction to ensure no disk clutter.
