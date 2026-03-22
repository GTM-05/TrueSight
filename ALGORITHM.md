# TrueSight "Strong Accurate Algorithm"

This document details the multi-stage forensic pipeline used to detect deepfakes and AI-generated content with high precision and low false-positive rates.

## 1. Multi-Stage Pipeline Overview

The engine follows a **Cascading Forensic Filter** approach:

### Stage 1: Fast Heuristics (Triage)
- **Laplacian Variance**: Detects unnatural blur or softening often used to hide synthesis artifacts (`Threshold: < 80`).
- **Metadata Inspection**: Checks for C2PA, EXIF, and specific AI-provider signatures (e.g., OpenAI, DALL-E).

### Stage 2: AI-Vision Analysis (ViT)
- **Vision Transformer (ViT)**: Analyzes sampled frames for high-frequency noise patterns characteristic of diffusion and GAN models.
- **Adaptive Routing**: Only the "most suspicious" frames (based on Stage 1) undergo heavy deep analysis to optimize performance.

### Stage 3: Spatio-Temporal Consistency
- **SSIM Sequence**: Measures structural similarity between consecutive frames. Sudden micro-fluctuations (`Std > 0.08`) indicate frame-by-frame synthesis.
- **Optical Flow (Farneback)**: Detects motion warping on facial regions. Unphysical pixel "sliding" marks a forgery.

### Stage 4: Biological Liveness (rPPG & Blinks)
- **Remote Photoplethysmography (rPPG)**: Analyzes the Green-channel chrominance of facial skin for human heart-rate periodicity using FFT.
- **Micro-Blink Detection**: Monitors rapid intensity dips in eye regions. A genuine human typically blinks at least 2 times per 10s sample.

### Stage 5: Structural Grid Detection (Elite FFT)
- **Elite Spectral Fallback**: Uses Fast Fourier Transform (FFT) on image gradients to find repeating "grid" artifacts left by upscalers and generation grids. 
- **Peak/Mean Ratio**: A ratio `> 120` definitively marks the presence of a synthetic structural grid.

## 2. Low-Confidence Capping (Safety Floor)

To prevent legitimate human videos from being misclassified due to environmental noise (e.g., background silence or compression blur), the algorithm uses a **Safety Floor** logic:

- **Condition**: If `Score < 75%` AND no **Strong Evidence** is found.
- **Strong Evidence Definition**:
    - AI Synth Probability `> 40%`
    - Structural Grids detected
    - Clear Biological Liveness Anomaly (Pulse failure with sufficient data)
- **Result**: The risk is capped at **19% (Low Risk)** if no strong markers are present.

## 3. Tuned Forensic Thresholds (ForensicConfig)

| Parameter | Value | Purpose |
|---|---|---|
| `AI_SYNTH_STRONG_THRESHOLD` | 45 | Probability > 45% is considered a definitive AI signature. |
| `AI_SYNTH_EVIDENCE_THRESHOLD` | 40 | Minimum probability to count as "Strong Evidence" for the safety floor. |
| `SAFETY_CAP_SCORE_LIMIT` | 75 | Scores below 75% without strong evidence are capped. |
| `SAFETY_CAP_RESULT` | 19 | The final score for videos that pass the safety floor (Low Risk). |
| `LIVENESS_CONFIRMED_FACTOR` | 0.1 | 90% risk reduction when full human liveness is confirmed. |
| `LIVENESS_PARTIAL_FACTOR` | 0.7 | 30% risk reduction for partial liveness (e.g., 1 blink). |
| `MIN_LIVENESS_SIGNALS` | 15 | Minimum skin chrominance samples for rPPG pulse FFT. |
| `MIN_LIVENESS_FACES` | 3 | Minimum distinct face detections needed for liveness. |
| `BLINK_MIN_HUMAN` | 2 | Target eye-blinks per 10s sample for genuine humans. |
| `LACK_OF_DATA_PENALTY` | 15 | Small penalty applied when biometric scan fails due to lack of faces. |
| `LAPLACIAN_BLUR_THRESHOLD` | 80 | Threshold for detecting synthesized blur/softening. |

## 4. Final Risk Computation
The final score is a weighted max of all sub-detectors, ensuring that a **single definitive fail** (e.g., a structural grid with Peak/Mean > 120) cannot be smoothed out by other passing metrics. 

### Verdict Logic:
- **High Risk (≥ 60%)**: Strong evidence of AI generation or manipulation.
- **Medium Risk (30–59%)**: Accumulated noise or weak AI signals.
- **Low Risk (< 30%)**: Verified human content or negligible artifacts.
