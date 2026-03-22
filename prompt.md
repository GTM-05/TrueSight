# TrueSight — Maintainer DNA & onboarding (v3.0)

Use this when extending or debugging **TrueSight**: a **local, multimodal, heuristic-first** forensic UI built on Python, Streamlit, OpenCV, librosa, and optional **Ollama**.

---

## Core identity

- **Forensic scores are not LLM outputs.** `modules/*` produce evidence; **`fusion/engine.py`** fuses them using **`config.ForensicConfig` (`CFG`)**. The LLM (`llm/llm.py`) only turns **structured facts** into prose.
- **Explainability** — Prefer tagged **`reasons`** (e.g. `[ELA]`, `[PHASE]`, `[FACE-WARP]`) so fusion, PDFs, and grouped UI lines stay traceable.
- **Privacy** — Default path is offline; URL analysis is the main “network” surface for the target URL itself, not for sending user media to third parties.

---

## Architecture snapshot

| Piece | Role |
|-------|------|
| `config.py` | Single dataclass of thresholds, fusion weights, morphing fusion, LLM defaults |
| `modules/image.py` | Image + **video frame** analysis; **different ELA/SRM/copy-move gates** when `source="video"` |
| `modules/video.py` | Sampling, liveness, **face SSIM morphing**, **face warp**, audio extract, **`compute_morphing_score`** |
| `modules/audio.py` | Audio detectors; **`sub_scores`** e.g. phase **`spike_count`** for morphing |
| `fusion/engine.py` | `compute_final_score`, `compute_morphing_score`, `generate_final_verdict_ai` |
| `llm/llm.py` | JSON brief + Ollama generate; fallback text if unavailable |
| `app.py` | Streamlit, **`display_indicators`**, session fusion + report |
| `verify_accuracy.py` | Regression smoke / optional strict benchmarks |

---

## Fusion v3.0 (mental model)

1. **Fuse** (strong-anchor + capped weak boost, or weighted blend).
2. **Cross-modal penalty** for diverging high-confidence scores.
3. **Cross-Detector Consensus (CDC)** — 4+ firing sectors = High Risk anchor.
4. **Liveness-gating** — Skip reduction if structural signals (ELA, AI Gen) are present.
5. **Safety floor** (graduated).

**Morphing** is a **parallel index** (face temporal + metadata + phase spikes), not a duplicate of raw video aggregate risk.

---

## Developer rules

1. **No magic numbers in detectors** — Add fields to **`ForensicConfig`** and read **`CFG`**.
2. **Preserve modality contracts** — `score`, `confidence`, `is_strong`, `reasons`, `sub_scores` / `metrics` as expected by fusion.
3. **Video frames** — Always thread **`source="video"`** through `analyze_image` for sampled frames.
4. **UI** — When showing long `reasons` lists, prefer **tag grouping** (`display_indicators`) for scanability.
5. **LLM changes** — Keep prompts tied to **engine numbers**; avoid dumping hundreds of raw duplicate lines into the model context.

---

## Stack

- **UI:** Streamlit  
- **Vision:** OpenCV, optional scikit-image SSIM inside face ROIs  
- **Audio:** librosa, scipy  
- **ML:** transformers + torch (ViT-style image path)  
- **Reports:** ReportLab (`reports/generator.py`, `llm/report_generator.py`)  
- **Narration:** Ollama Python client; model name from **`CFG.LLM_VERDICT_MODEL`**

---

## Operational workflow

1. User runs one or more tabs (image / audio / video / URL).  
2. Results land in `st.session_state`.  
3. **Generate Final Forensic Report** builds `all_evidence` and calls **`generate_final_verdict_ai`** (unless Turbo / `skip_llm`).  
4. Optional PDF from tab-specific actions or external `generate_report` helpers.

---

## Quality bar

- **Accurate docs** — README / ARCHITECTURE / PROJECT_SPEC / ALGORITHM / SETUP should match `config.py` and `fusion/engine.py`, not legacy v1 fusion math.
- **Small diffs** — Fix the narrowest layer (config vs module vs fusion) for a given bug.
