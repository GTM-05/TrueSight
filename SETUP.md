# TrueSight setup guide (v3.0)

Deploy a **local** environment: FFmpeg on PATH, Python **venv**, optional **Ollama** for narrative (configurable model).

---

## 1. Prerequisites

- **OS:** Linux (Ubuntu 22.04+ recommended)  
- **RAM:** 8 GB comfortable; 4 GB possible with **Low Resource Mode** and smaller LLM  
- **Disk:** ~5–10 GB for Python deps, ViT weights, and Ollama models  

### System packages

```bash
sudo apt update
sudo apt install -y ffmpeg python3 python3-venv
```

`ffmpeg` / `ffprobe` are required for video audio extraction and metadata.

---

## 2. Ollama (optional but recommended for reports)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve   # background terminal or systemd
```

Pull a model that matches **`config.py`** → **`LLM_VERDICT_MODEL`** (default **`qwen2.5:3b`**). Examples:

```bash
ollama pull qwen2.5:3b
# lighter alternative:
ollama pull qwen2:0.5b
```

If you use a non-default tag, either change **`CFG.LLM_VERDICT_MODEL`** or pass a model override where the app supports it.

Verify:

```bash
curl -s http://localhost:11434/api/tags | head
```

---

## 3. Python environment

```bash
cd TrueSight
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If the system has no `pip` for `python3`, the venv step still provides `pip` inside `.venv`.

---

## 4. Run the app

```bash
source .venv/bin/activate
streamlit run app.py
```

- **Sidebar → Low Resource Mode** — Fewer frames, lighter path.  
- **Sidebar → Turbo Report (Instant)** — Skips LLM; fusion summary only.  
- **Deep Scan** — More video samples when enabled.

---

## 5. Smoke / regression tests

```bash
source .venv/bin/activate
python verify_accuracy.py
```

Add labeled clips under `test_samples/` (see script docstring). Use **`--strict-benchmarks`** for pass/fail bands on `Showcase_*` files.

---

## 6. Troubleshooting

| Issue | What to check |
|-------|----------------|
| Ollama / LLM errors | `ollama serve` running; model pulled; `LLM_VERDICT_MODEL` matches local tags |
| FFmpeg errors | `ffmpeg -version`, `ffprobe -version` |
| Import errors | Active **`.venv`**, `pip install -r requirements.txt` |
| CUDA / torch | CPU-only installs work; GPU optional |

---

## 7. Configuration

Edit **`config.py`** (`ForensicConfig`) for thresholds, fusion, morphing fusion, and LLM defaults. Restart Streamlit after changes.

---

## 8. Credits

Maintainer: **GTM-05** — see repository for license / use terms.
