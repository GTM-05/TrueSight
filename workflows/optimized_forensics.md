# TrueSight — Operator workflow notes (v3.0)

Practical workflow for **accurate** results without unnecessary load. This doc matches the **current** `app.py` tabs and engines (not legacy `app-mini` / `app-ai` names).

---

## 1. Triage mindset

TrueSight separates:

- **Aggregate modality risk** (image / audio / video / URL scores fused in `fusion/engine.py`)  
- **Morphing / manipulation index** (video spatial + metadata + audio phase spikes)  
- **AI synthesis hints** (e.g. video `ai_gen_score`, image ViT path)  
- **Threat scan** (binary / malware-style score from `threats.py`)

Use **all** relevant tabs for the artifact you care about; the **final report** only fuses modalities that have been run in the session.

---

## 2. Recommended order

1. **Malware / threats** — Run upload analysis; if the threat score is non-zero, treat the file as a security incident first.  
2. **Primary modality** — Video file → **Video** tab; still image → **Image**; clip with suspicious audio → **Audio** too.  
3. **URL** — If the case is link-centric, run **URL** separately.  
4. **Final forensic report** — After populating session results, use **Generate Final Forensic Report**. Leave **Turbo** off if you want the LLM narrative (requires Ollama + pulled model).  

---

## 3. Resource modes

| Mode | When to use |
|------|----------------|
| **Default** | Normal laptops / workstations |
| **Low Resource** | RAM- or CPU-constrained hosts; fewer frames, lighter checks |
| **Deep Scan** | Higher frame count for video when you accept longer runtime |
| **Turbo Report** | Instant deterministic summary; **no** Qwen narrative |

---

## 4. Reading indicators

- Raw analyzer output can emit **many** similar lines (e.g. per-frame `[ELA]`). The **video** tab uses **`display_indicators()`** to **group by tag** for readability.  
- **Fusion expander** lists mathematical **key_findings** from merged `reasons` (trimmed).  

For PDFs, tab-specific “Generate … Report” buttons use `reports/generator.py`; optional alternate PDF flow lives under `llm/report_generator.py`.

---

## 5. Validation

- **`python verify_accuracy.py`** — Ensures video + fusion path runs on sample MP4s.  
- **`--strict-benchmarks`** — Use when you maintain golden clips in `test_samples/`.  

---

## 6. Future optimizations (not required today)

Ideas compatible with the current architecture:

- **Parallelism** — Metadata + threat scan in a thread pool while the heavy modality runs (would need careful temp-file discipline).  
- **Content-hash cache** — Skip re-analysis of identical uploads within a TTL.  
- **LLM** — Already streams in the UI when not in Turbo mode; narrative quality scales with model size (`CFG.LLM_VERDICT_MODEL`).

---

## 7. Human-in-the-loop

Heuristic and ML detectors produce **investigative** signals, not legal proof. For high-stakes cases, corroborate with **source acquisition**, chain of custody, and expert review. Use TrueSight outputs as structured starting points, not sole evidence.
