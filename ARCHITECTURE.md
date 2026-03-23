# TrueSight — Architecture (v3.0)

Privacy-first, local multimodal forensics: **detectors → fusion math → optional LLM narrative → PDF/UI.**

---

## 1. Philosophy

- **Heuristic-first** — Prefer explainable signal processing and bounded ML (ViT) over opaque cloud APIs.
- **Single source of truth for risk** — `fusion/engine.py` + `config.py`; the LLM does not vote on scores.
- **Modality contracts** — Each analyzer returns `score`, `confidence`, `is_strong`, `reasons`, and usually `sub_scores` / `metrics` for downstream fusion and reporting.

---

## 2. Analysis layer (`modules/`)

| Module | Responsibility |
|--------|----------------|
| **`image.py`** | Image and **extracted video frame** analysis; ELA and several stats use **`source="video"`** branches where codec noise differs from standalone files. |
| **`video.py`** | Frame sampling, temp files, per-frame `analyze_image(..., source="video")`, rPPG / liveness block, optional lip–audio correlation, **`detect_ssim_morphing`**, **`detect_face_warp`**, metadata, WAV extract + **`analyze_audio`**, then **`compute_morphing_score`** for **`morphing_score`**. |
| **`audio.py`** | Pitch, MFCC, phase discontinuities, HNR, etc.; **`sub_scores`** include structured entries (e.g. phase **`spike_count`**) for morphing fusion. Confidence floor refined to 0.3 for synthetic speech detection (v3.1). |
| **`url.py`** | URL threat / trust heuristics. |
| **`metadata.py`** | EXIF / container hints. |
| **`threats.py`** | Binary / malware-style scan for uploads. |

---

## 3. Fusion layer (`fusion/engine.py`)

**`compute_final_score(image, audio, video, liveness, url?)`** runs:

1. Fuse modality dicts (strong-anchor + bounded weak boost, or weighted blend). Confident strong signals (score >= 65) anchor the decision.
2. Cross-modal penalty if confident modalities disagree strongly.
3. **Cross-Detector Consensus (CDC)** — Boost and anchor score if 4+ independent forensic sectors fire simultaneously.
4. **Liveness-gating** — Reduction is **skipped** if structural signals (ELA, AI Gen, etc.) are detected, preventing spoofing.
5. Graduated safety floor and reason append.

**`compute_morphing_score(video_result, audio_result)`** is a **separate index** (face morph + metadata + phase spikes) used for manipulation/morphing metrics and LLM briefs, not a replacement for the main fusion formula.

**`generate_final_verdict_ai(all_evidence, ...)`** maps Streamlit session keys `Image` / `Audio` / `Video` / `URL` into the above, attaches **`morphing_score`** from video, and calls **`llm_generate_explanation`** with the **fusion dict** unless `skip_llm` (Turbo).

---

## 4. Configuration (`config.py`)

All major thresholds and LLM defaults live in **`ForensicConfig`** (instance **`CFG`**). Application code should read **`CFG`** instead of scattering literals.

---

## 5. Presentation layer (`app.py`)

- Sidebar: low resource mode, deep scan, **Turbo Report** (`skip_llm`).
- Per-tab analysis populates `st.session_state.*_result`.
- **Final report** builds `all_evidence` and calls **`generate_final_verdict_ai`**; may stream LLM tokens.
- **`display_indicators(reasons)`** collapses repeated `[TAG]` lines for readable warnings.

---

## 6. LLM layer (`llm/llm.py`)

- Builds a **JSON “forensic brief”** from session evidence + **fusion** output (scores, modality list, trimmed top reasons, video SSIM / morph fields).
- System prompt instructs the model to **not contradict** numeric fields and to mark missing modalities as not analyzed.
- Uses **`CFG.LLM_VERDICT_*`** for model name, temperature, token limit; falls back to a **deterministic** summary if Ollama fails or the reply is too short.

**`llm/report_generator.py`** is a secondary ReportLab path with its own shorter prompt; **`reports/generator.py`** drives the main UI PDF dossier.

---

## 7. Data lifecycle (typical video)

1. Upload → temp file on disk.  
2. `cv2` sample frames → JPEG paths → `analyze_image` per frame.  
3. Face ROIs → rPPG + eye stats → liveness dict.  
4. FFmpeg → WAV → `analyze_audio`.  
5. Morphing SSIM/warp on video path + indices.  
6. Aggregate + reasons → session state → fusion / final report.  
7. Temp frames and audio deleted under `tempfile` cleanup.

---

## 8. Design patterns

- **Reason tags** — Strings like `[ELA]`, `[PHASE]` keep fusion and UI grouping consistent.
- **Streaming LLM** — Optional for responsiveness; same brief as non-streaming.
- **Session cache** — Final verdict cached on a coarse evidence hash (see `app.py`).
