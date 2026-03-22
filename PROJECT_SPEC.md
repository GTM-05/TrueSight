# TrueSight — Project reference (v3.0)

## Overview

TrueSight is a **local, offline-first** multimodal forensic assistant. **Numeric risk is produced by deterministic code** (`modules/*`, `fusion/engine.py`, `config.py`). **Ollama / Qwen** is used only to turn structured outputs into prose (and can be bypassed with “Turbo Report” in the UI).

---

## Fusion engine (`fusion/engine.py`)

Version 3.0 uses a **fixed four-step pipeline** (not the legacy weighted max/avg formula):

3. **Cross-Detector Consensus (CDC)** — If 4+ independent sectors (AI, Metadata, Audio, Noise, etc.) fire, the score is boosted and anchored at High Risk (>=85%) or the `CONSENSUS_SCORE_FLOOR`.
4. **Liveness-gating** — If any **structural** signals (ELA, SRM, AI, Morph) fire, liveness-based reduction is **skipped** to prevent metabolic spoofing.
5. **apply_safety_floor** — Graduated floors for consistent evidence tracking.

Outputs include **`verdict`** (`HIGH` / `MEDIUM` / `LOW RISK` strings), **`sub_scores`** per modality, merged **`reasons`**, and flags such as **`liveness_detected`**.

**Morphing index (video-centric):** `compute_morphing_score(video_result, audio_result)` blends face SSIM morph score, face warp, scaled metadata score, and **phase spike_count** from audio `sub_scores`. The **final multimodal verdict** exposes **`morphing_score`** from the video payload after fusion (see `generate_final_verdict_ai`).

---

## Verdict bands (`ForensicConfig`)

| Final score | UI / short label |
|-------------|------------------|
| ≥ `HIGH_RISK_THRESHOLD` (60) | High |
| ≥ `MEDIUM_RISK_THRESHOLD` (30) | Medium |
| Below 30 | Low |

---

## Config highlights (`config.py` → `CFG`)

Single dataclass **`ForensicConfig`** holds tunables. Examples:

| Area | Examples | Role |
|------|-----------|------|
| Image ELA | `ELA_MEAN_THRESHOLD`, `ELA_STD_THRESHOLD` | Standalone JPEGs / PNGs |
| Video frames | `ELA_MEAN_THRESHOLD_VIDEO`, `ELA_STD_THRESHOLD_VIDEO` | FFmpeg frames run hotter — separate gates |
| Face SSIM | `SSIM_FACE_*`, `SSIM_FACE_MORPH_*` | Dual pathology: too-stable vs too-variable |
| Face warp | `OPTICAL_FLOW_WARP_THRESHOLD`, `FACE_WARP_*` | ROI optical flow |
| Morphing fusion | `MORPHING_META_*`, `MORPHING_PHASE_*`, `MORPHING_SPATIAL_WEIGHT` | Unified morph index |
| rPPG | `RPPG_SNR_MIN`, `RPPG_SNR_ANOMALY`, `MIN_LIVENESS_*` | Pulse reliability guards |
| LLM | `LLM_VERDICT_MODEL`, `LLM_VERDICT_NUM_PREDICT`, `LLM_VERDICT_MIN_CHARS` | Ollama narrative |

Adjust **only `config.py`** for thresholds unless you are changing detector logic.

---

## Scores shown in the UI

- **Total video risk** — Aggregated frame + rPPG + sync + metadata reasons (not the same as morphing-only).
- **AI synthesis (video)** — `ai_gen_score` (e.g. percentile of frame-level scores).
- **Morphing** — `morphing_score` / manipulation index from spatial + metadata + phase.
- **Final forensic report** — `generate_final_verdict_ai` merges session modalities; **`ai_generated_score`** uses max of image score, audio score, and video **`ai_gen_score`** (or video aggregate if synthesis score unset).

---

## Dependencies (typical)

`streamlit`, `torch`, `transformers`, `opencv-python`, `numpy`, `scipy`, `librosa`, `scikit-image`, `Pillow`, `tldextract`, `exifread`, `reportlab`, `ollama` (Python client), `ffmpeg` / `ffprobe` on PATH.

---

## Related files

- **`verify_accuracy.py`** — Smoke / optional strict benchmarks on `test_samples/*.mp4`.
- **`reports/generator.py`** — Premium dossier PDF from tab-specific dicts.
- **`llm/report_generator.py`** — Alternative compact PDF that can call Ollama with a simpler prompt.
