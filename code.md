# TrueSight — Complete Code Flow Reference

This document explains every file, what it does, how data flows through it, and why each piece exists.

---

## Entry Point: `app.py`

**What it does:** The entire Streamlit UI. Contains 4 analysis tabs + the report section.

**Flow:**
```
User uploads file
     │
     ▼
Tab 1 (Video)  → scan_for_threats() → analyze_video()  → st.session_state.video_result
Tab 2 (Image)  → scan_for_threats() → analyze_image()
                                    → check_metadata()  → st.session_state.image_result
Tab 3 (Audio)  → scan_for_threats() → analyze_audio()  → st.session_state.audio_result
Tab 4 (URL)                         → analyze_url()    → st.session_state.url_result
     │
     ▼
"Generate Report" button
     │
     ▼
all_evidence = { 'Video': ..., 'Image': ..., 'Audio': ..., 'URL': ... }
     │
     ▼
generate_final_verdict_ai(all_evidence) → returns verdict dict
     │
     ▼
Display scores + Qwen2 explanation + Download PDF
```

**Key Variables:**
- `low_res` (bool) — sidebar checkbox, reduces frame count to 1 and skips SSIM
- `all_evidence` (dict) — assembled from session state before report generation
- `verdict` (dict) — returned from fusion engine with all scores + ai_explanation

---

## `modules/image.py` — Image Forensics

**What it does:** Detects AI-generated images using two independent methods.

**Flow:**
```
analyze_image(path, low_resource)
     │
     ├── detect_ai_generated(path)
     │       ├── [If model loaded] → ViT pipeline (jacoballessio/ai-image-detect-distilled)
     │       │     → Resizes to 224×224 → classifies "artificial" vs "human"
     │       │     → Returns ai_probability (0–100)
     │       └── [Fallback] → _heuristic_ai_detection(path)
     │               → Checks resolution against known AI output sizes
     │               → FFT frequency smoothness (AI images are unnaturally smooth)
     │               → Returns ai_probability (max 55% — limited heuristic)
     │
     ├── error_level_analysis(path)
     │       → Saves JPEG at q=90, compares with original
     │       → High ELA std = editing artifacts = suspicious
     │       → Returns ela_map_path for visualization
     │
     └── Score assembly:
             ai_prob ≥ 80% → +80 pts (CRITICAL)
             ai_prob ≥ 50% → +55 pts
             ai_prob ≥ 35% → +35 pts
             ela_std > 15  → +20 pts
             ela_std > 8   → +10 pts
             → final score = min(100, total)
```

**Why ViT + ELA?**
- ViT catches AI generation patterns in semantic features (the "style")
- ELA catches post-processing artifacts (compression inconsistencies from editing)
- Together they cover both "was this generated" AND "was this edited"

---

## `modules/audio.py` — Audio Forensics

**What it does:** Detects synthetic/cloned voices using signal processing.

**Flow:**
```
analyze_audio(path)
     │
     ├── librosa.load(path, sr=None)   ← sr=None preserves native rate (no resampling)
     │
     ├── MFCC Analysis
     │       mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)
     │       mfcc_delta = librosa.feature.delta(mfcc)
     │       Low mfcc_delta_std → TTS voices have unnaturally smooth phoneme transitions
     │
     ├── Pitch Analysis (via librosa.yin)
     │       Extract fundamental frequency (F0)
     │       Low pitch_std → robotic monotone (TTS characteristic)
     │
     ├── RMS Energy (Loudness)
     │       rms_std → TTS voices have flat, consistent energy (no natural breath variation)
     │
     ├── Spectral Features
     │       spectral_centroid → brightness
     │       spectral_rolloff  → frequency distribution skew
     │       spectral_flatness → noise-like vs tonal (very flat = synthetic)
     │
     └── IsolationForest anomaly scoring on combined features
             → Returns score 0–100
```

**Why these features?**
- Real voices are **noisy** — breath, rhythm, stress all create variation
- TTS voices optimize for clarity — they are too perfect, too flat
- The combination of pitch_std + mfcc_delta_std is the strongest signal

---

## `modules/video.py` — Video Forensics

**What it does:** Multi-layer video forensics using visual, temporal, acoustic, and metadata analysis.

**Flow:**
```
analyze_video(path, low_resource)
     │
     ├── Frame Extraction (cv2)
     │       Samples 5 frames (or 1 if low_resource=True)
     │       Saves to tempfile directory
     │
     ├── Per-Frame AI Detection
     │       For each frame → analyze_image(frame_path)
     │       Collects ai_frame_scores = [s1, s2, ...]
     │
     ├── Multi-Frame Consistency Bonus
     │       If 3+ frames all score ≥ 40% → +20 bonus
     │       If 2+ frames all score ≥ 40% → +10 bonus
     │       (Real videos rarely have consistent AI artifacts across time)
     │
     ├── SSIM Temporal Consistency (skipped in low_resource)
     │       Compares consecutive frames for structural similarity
     │       High std in SSIM → abrupt structural changes → deepfake splice
     │
     ├── Audio Extraction (ffmpeg)
     │       ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 audio.wav
     │       → analyze_audio(audio.wav)
     │       If NO audio track exists → +60 pts (AI raw output has no audio)
     │
     ├── Metadata Check (ffprobe)
     │       ffprobe -print_format json -show_format -show_streams video.mp4
     │       Missing/sparse tags → +35 pts (AI generators leave no metadata)
     │       Known AI encoder strings → +90 pts
     │
     └── Smart Weight Redistribution
             If audio + meta available:  visual×0.6 + audio×0.2 + meta×0.2
             If only audio:              visual×0.75 + audio×0.25
             If only real meta (≥20):    visual×0.75 + meta×0.25
             If neither available:       visual×1.0 (trust what we have)
```

**Why not MoviePy?**
- cv2 (C++ bindings) extracts frames in milliseconds vs MoviePy's Python overhead
- MoviePy loads the entire video into RAM; cv2 is frame-by-frame

---

## `modules/url.py` — URL Forensics

**What it does:** Detects phishing, homograph attacks, DGA domains, and URL shorteners.

**Flow:**
```
analyze_url(url)
     │
     ├── tldextract → domain, subdomain, TLD
     ├── Shannon entropy of domain → high entropy = DGA / randomly generated
     ├── Homograph detection → Unicode chars that look like ASCII (е vs e)
     ├── URL shortener check → bit.ly, t.co, goo.gl etc
     ├── Subdomain depth (> 3 levels suspicious)
     ├── Suspicious keyword match → 'login', 'verify', 'secure', 'paypal', 'bank'
     └── Score assembly → 0–100
```

---

## `modules/metadata.py` — EXIF + Video Tags

**What it does:** Reads embedded metadata from image EXIF and video container tags.

**Image flow:** `exifread.process_file()` → checks for editing software tags (Photoshop, GIMP), missing GPS, suspicious creation dates

**Video flow:** `ffprobe -print_format json` → reads `format.tags` → checks encoder, creation_time, sparse tags

---

## `modules/threats.py` — Malware Detection

**What it does:** Quick file-level threat scan before any AI analysis.

**Checks:**
- File extension vs actual MIME type mismatch (e.g., `.mp4` that is actually an `.exe`)
- High entropy content (possible encrypted/packed malware)
- Known malicious byte signatures (magic bytes)

---

## `fusion/engine.py` — Decision Layer

**What it does:** The final mathematical verdict. **No LLM involved here.**

**Flow:**
```python
img_score  = all_evidence['Image']['score']   # 0–100
aud_score  = all_evidence['Audio']['score']   # 0–100
vid_score  = all_evidence['Video']['score']   # 0–100
url_score  = all_evidence['URL']['score']     # 0–100

final_score = (0.35 * img_score) + (0.25 * aud_score) +
              (0.25 * vid_score) + (0.15 * url_score)

verdict  = 'High' if final_score >= 60 else 'Medium' if >= 30 else 'Low'
confidence = abs((final_score / 100.0) - 0.5) * 2 * 100

→ calls llm_generate_explanation() for the narrative only
```

**Why these weights?**
- Image (35%): Most direct evidence — ViT model is highly reliable on still frames
- Video (25%): Multi-frame + audio + metadata = rich but takes longer
- Audio (25%): Strong synthetic voice detector via spectral features
- URL (15%): Supporting context only — many malicious URLs are legitimate-looking

---

## `llm/llm.py` — Explanation Layer

**What it does:** Qwen2 (0.5B) writes the forensic explanation after the verdict is decided.

**Critical rules enforced in the prompt:**
```
- You DO NOT decide whether the content is fake or real.
- The verdict is already given — you must only justify it.
- Do NOT hallucinate. Base explanation only on numerical evidence.
```

**Flow:**
```
llm_generate_explanation(evidence, final_score, verdict, confidence)
     │
     ├── Extracts numeric features from evidence dict
     ├── Builds structured prompt with all scores
     ├── ollama.generate(model='qwen2:0.5b', temperature=0.1, max_tokens=400)
     └── Returns structured text:
             Summary: (2–3 lines)
             Technical Analysis: (evidence-based, including Optical Flow & ViT)
             Conclusion: (verdict reinforcement)
```

**Why temperature=0.1?** Low temperature = deterministic, factual output. We do not want creativity in forensic reports.

---

## `reports/generator.py` — PDF Export

**What it does:** Generates a professional forensic dossier PDF using reportlab.

**Sections generated:**
1. Case metadata (timestamp, file analyzed)
2. Per-modality score breakdown
3. Key findings list
4. Qwen2 (0.5B) forensic explanation narrative
5. Confidence rating + verdict stamp

---

## Data Flow Summary

```
Upload → [Threat Scan] → [Module Analysis] → session_state
                                                   │
                                     "Generate Report" button
                                                   │
                                          all_evidence dict
                                                   │
                                       fusion/engine.py
                                      (maths → verdict)
                                                   │
                                         llm/phi3.py
                                      (words → narrative)
                                                   │
                                    reports/generator.py
                                       (PDF download)
```
