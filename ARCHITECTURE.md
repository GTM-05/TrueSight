# TrueSight Architecture Decision Record

This document outlines the system design and architectural patterns for TrueSight AI.

## 1. System Philosophy
TrueSight is designed to be **Privacy-First** and **Resource-Efficient**. It runs entirely on local hardware (optimized for 8GB RAM) and uses specialized forensic heuristics to augment small-scale AI models.

## 2. Core Modules

### 2.1 Analysis Layer (`modules/`)
These modules extract raw forensic evidence without using the LLM.
- **`video.py`**: Orchestrates **CHROM-rPPG** liveness, **Iris Jitter** (Gaze Naturalness), and SSIM consistency.
- **`image.py`**: Performs **Radial Spectral Slope** analysis, **Chromatic Alignment**, and ELA re-compression checks.
- **`audio.py`**: Performs signal processing (Pitch std, MFCC, Flux) to catch voice clones.
- **`metadata.py`**: Cross-references internal file tags with known generative AI tool signatures.

### 2.2 Fusion Intelligence (`fusion/`)
- **`engine.py`**: Implements the **Max-Biased Fusion** logic. It takes raw scores and biological overrides (rPPG/Iris) to calculate the final verdict, applying the **19% Safety Floor** for low-confidence data.

## 3. Data Flow (Lifecycle of an Analysis)

1.  **Ingestion**: Streamlit UI accepts a multi-modal file upload.
2.  **Artifact Extraction**: `image.py` or `video.py` performs spectral and structural scans.
3.  **Biological Triage**: Face tracking isolates ROI -> CHROM-rPPG verifies heart rate -> Iris Jitter verifies gaze.
4.  **Cascading Fusion**: Results are vectorized. A "Smoking Gun" (e.g., Grid Artifact) overrides broad averages.
5.  **Narration**: LLM receives a context string (Scoring Table + Evidence list) and generates the "Dossier."
6.  **Archival**: `reports/generator.py` serializes the result into a timestamped PDF.

## 4. Key Architectural Patterns

- **Cascading Precision**: Expensive ML models (ViT) are only run on a subset of "Candidate Frames" to preserve RAM.
- **The 19% Safety Floor**: A design pattern that prevents "False Positive Creep" by capping scores that lack definitive forensic markers.
- **Stateless Analysis**: Every request is isolated. The `tempfile` module is used for frame extraction to ensure no disk clutter.
