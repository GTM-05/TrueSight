# TrueSight Forensic Algorithm: "Strong Accurate"

The "Strong Accurate" algorithm is a cascading multi-modal forensic filter designed to distinguish between human content, traditional deepfakes, and next-gen AI generations (e.g., Sora, FLUX).

## 1. The Multi-Stage Pipeline

Analysis follows a tiered approach to ensure both speed and precision:

1.  **Stage 1: Fast Heuristics** (Laplacian Variance, Resolution Checks, Format tags).
2.  **Stage 2: Artifact Extraction**:
    - **ELA (Error Level Analysis)**: Detects differential re-compression.
    - **FFT Grid Detection**: Identifies mathematical lattices left by per-frame synthesis.
3.  **Stage 3: Elite Signal Analysis** (New in Phase 2):
    - **Radial Spectral Slope**: Checks if log-power spectrum follows the **1/f² law**. AI models often create unnatural frequency distributions.
    - **Chromatic Alignment**: Detects suspiciously perfect channel alignment typical of synthetic sensors.
4.  **Stage 4: Biological Verification**:
    - **CHROM-rPPG**: Chrominance-based pulse extraction to verify biological liveness.
    - **Iris Jitter**: Detection of micro-saccade patterns to verify gaze naturalness.

## 2. Decision Logic

### 2.1 The 19% Safety Floor
To prevent "Death by a Thousand Cuts" (where minor noise results in a False Positive), any sample lacking a **definitive forensic anchor** is capped at **19% (Low Risk)**. 
Anchors include:
- High-confidence ViT score (> 0.5)
- Structural Grid artifacts detected
- Biological Pulse Anomaly (rPPG SNR < threshold)

### 2.2 Max-Biased Fusion
TrueSight does not use simple averaging. It uses **Max-Biased Fusion**:
```python
raw_score = (max_module_score * 0.9) + (weighted_avg * 0.1)
```
This ensures that if a single module finds a "smoking gun" (e.g., 90% risk in Audio), the final result reflects that danger even if other modules (Metadata, URL) are clean.

## 3. Heuristic Thresholds

| Metric | Normal Range | Suspicious Range |
|---|---|---|
| Spectral Slope | -2.2 ± 0.2 | > ±0.5 deviation |
| Chromatic Align | > 3.0 | < 2.5 (Too Perfect) |
| rPPG Pulse SNR | > 1.5 | < 0.8 (Anomaly) |
| Iris Jitter | Dynamic | Static/Locked |

## 4. Operational Flow
The final risk level is calculated by combining:
- **AI-Generated Score**: Synthesis confidence.
- **Manipulation Score**: Edit/Flicker/Sync artifacts.
- **Threat Score**: Malicious code/binary detection.
