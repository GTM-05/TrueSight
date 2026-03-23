# TrueSight forensic algorithm (v3.0)

Cascading **multi-modal** pipeline: fast metadata and heuristics, then deeper image/audio/video signals, then **fusion** and optional **narration**.

---

## 1. Pipeline stages

### Image (and video frames)

1. **Quick checks** — Resolution heuristics, optional ViT-style probability, metadata hooks.
2. **ELA** — Error level analysis; **standalone images** use calibrated thresholds (`ELA_MEAN_THRESHOLD`=75.0 / `ELA_STD_THRESHOLD`=85.0) to accommodate high-quality human samples; **video frames** (`source="video"`) use more sensitive gates (**`ELA_MEAN_THRESHOLD_VIDEO`** / **`ELA_STD_THRESHOLD_VIDEO`**) to catch codec-transient artifacts.
3. **Spectral / chroma / noise / DCT** — Tuned for image vs video frame behavior. Spectral slope tolerance raised to 2.0 and DCT-GRID ratio to 250.0 to reduce false positives on human-capture images.
4. **Copy-move** — SIFT-based; higher match threshold for video frames (`COPY_MOVE_MIN_MATCHES_VIDEO`).

### Video

1. **Uniform frame sampling** — Count depends on low-resource / deep-scan flags.
2. **Per-frame image pipeline** — Each sample written as JPEG and analyzed with **`source="video"`**.
3. **Liveness** — CHROM-style rPPG on face ROI sequence; SNR guards (`RPPG_SNR_MIN`, `RPPG_SNR_ANOMALY`); blink proxy and iris jitter from ROI stats.
4. **Temporal face morphing** — **Adjacent face-crop SSIM** with **dual thresholds**: unnaturally stable (slow morph) vs high variance (splice/warp). Not full-frame SSIM.
5. **Face warp** — Farneback optical flow statistics inside face ROI vs `OPTICAL_FLOW_WARP_THRESHOLD` / `FACE_WARP_FLOW_STD_THRESHOLD`.
6. **Audio track** — FFmpeg mono 16 kHz WAV → shared **`analyze_audio`**; phase detector yields **spike_count** used in **`compute_morphing_score`**.
7. **Optional lip–audio** — Correlation between mouth aspect ratio sequence and envelope (non–low-resource paths).

### Audio (standalone or extracted)

- Pitch statistics (robotic jitter < 0.09), MFCC abrupt changes, harmonic-to-noise ratio, **unusually low spectral flatness** (< 0.015), **phase discontinuity** spikes (edit/splice sensitivity with 1.8 threshold), silence / TTS hints.

### URL

- Homograph, entropy, shorteners, suspicious TLDs, redirect parameters, optional IP handling.

---

## 2. Fusion decision logic (high level)

Not a single `max*0.9 + avg*0.1` formula. The engine:

- **Rewards strong modalities** and adds **limited** contribution from weak ones.
- **Cross-Detector Consensus (CDC)** — If 4+ independent sectors (e.g. AI, Metadata, Noise, ELA) fire, the score is boosted and anchored at High Risk (>=85%) to prevent subtle deepfakes from being missed.
- **Improved Confidence Gating** — `is_strong` requires > 0.3 confidence for audio following v3.1 refinement.
- **Penalizes** inconsistent high-confidence modalities (**cross-modal** spread).
- **Reduces** score when **liveness** supports a real recording, but this is **gated** by structural signals (ELAs, SRM, AI Gen) to avoid spoofing.
- Applies a **graduated safety floor** so weak evidence does not creep to high risk without anchors.

Verdict labels use **`HIGH_RISK_THRESHOLD`** (60) and **`MEDIUM_RISK_THRESHOLD`** (30).

---

## 3. Morphing index (separate from “video risk %”)

**`compute_morphing_score`** combines:

- Spatial: max of face SSIM morph score and face warp score × **`MORPHING_SPATIAL_WEIGHT`**
- Metadata: scaled/capped **`meta_score`**
- Audio: **`spike_count`** × **`MORPHING_PHASE_POINTS_PER_SPIKE`**, capped by **`MORPHING_PHASE_SCORE_CAP`**

This feeds **`morphing_score`** / manipulation metrics in the UI and LLM brief, while **aggregate video `score`** still reflects frame detectors, rPPG, sync, etc.

---

## 4. Heuristic reference (order of magnitude)

Exact numbers belong in **`config.py`**. Illustrative:

| Signal | Role |
|--------|------|
| Spectral slope | Compare radial power law to expected natural image slope |
| Chromatic shift | Too-perfect alignment vs extreme warp |
| rPPG SNR | Pulse reliability; low SNR with guards → anomaly reasons |
| Phase spikes | Dense discontinuities → splice / generation artifacts |
| ViT (video) | Frame agreement thresholds (`VIT_*_VIDEO`) reduce single-frame noise |

---

## 5. Threat and reporting

- **`threats.py`** contributes a separate **malware-style** score merged in the UI.
- **PDF** — `reports/generator.py` (dossier) and optionally `llm/report_generator.py`.

---

## 6. LLM role

**Does not compute risk.** It receives a structured brief (JSON) built from engine outputs and is instructed to stay consistent with those numbers. Turbo mode skips the LLM and prints a short deterministic summary.
