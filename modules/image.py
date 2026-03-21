"""
modules/image_ai.py — AI-Enhanced Image Analysis for app-ai.py

Adds a HuggingFace AI-image-detector on top of existing ELA + blur heuristics.
The transformer model detects whether an image was AI-generated (Midjourney, DALL-E,
Stable Diffusion etc.) — something ELA alone cannot do.

Falls back gracefully to heuristics-only if torch/transformers are not installed.
"""

import cv2
import numpy as np
import os
from PIL import Image

# Heavy imports moved inside _load_detector to save RAM in Ultra-Lite mode
import gc

# ── HuggingFace AI-image detector (loaded lazily on first call) ──────────────
_ai_detector = None
_detector_ready = False

def _load_detector():
    global _ai_detector, _detector_ready
    if _detector_ready:
        return _ai_detector
    try:
        import os
        # Force offline mode: skip all HuggingFace network calls (saves ~25s per run)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        # Use 'prithivMLmods/Deep-Fake-Detector-Model' - robust and stable for various library versions.
        _ai_detector = pipeline(
            "image-classification",
            model="prithivMLmods/Deep-Fake-Detector-Model",
            device=device,
            trust_remote_code=True
        )
        _detector_ready = True
        print("[IMAGE] ✅ ViT detector loaded from local cache (offline mode)")
    except Exception:
        print("[IMAGE] ⚠️ Offline: Using Elite FFT Spectral Fallback (AI Grid Detection)")
        _ai_detector = None
        _detector_ready = True
    return _ai_detector


# ── Face Detection (Booster for Deepfake Accuracy) ──────────────────────────
_face_cascade = None

def _get_face_detector():
    global _face_cascade
    if _face_cascade is None:
        try:
            # Standard Haar Cascade for front-facing faces
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            _face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass
    return _face_cascade


def detect_ai_generated(image_path: str, low_resource: bool = False, face_focus: bool = True) -> dict:
    """
    Uses a pre-trained ViT model to score the probability that an image
    was AI-generated. Returns a score 0–100 and a label.
    Falls back to heuristic indicators if model is unavailable.
    """
    detector = _load_detector()
    face_cascade = _get_face_detector()

    # ── ML path ──────────────────────────────────────────────────────────────
    # Skip ML path in Low Resource Mode (8GB RAM optimization)
    if not low_resource and detector is not None:
        face_metrics = {}
        img_cv = cv2.imread(image_path)
        img_target = None
        try:
            if img_cv is not None and face_focus and face_cascade is not None:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                full_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                    pad_w, pad_h = int(w * 0.15), int(h * 0.15)
                    x1, y1 = max(0, x-pad_w), max(0, y-pad_h)
                    x2, y2 = min(img_cv.shape[1], x+w+pad_w), min(img_cv.shape[0], y+h+pad_h)
                    face_crop = img_cv[y1:y2, x1:x2]
                    face_var = cv2.Laplacian(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    
                    face_metrics = {'face_var': float(face_var), 'bg_var': float(full_var)}
                    img_target = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    print(f"[IMAGE] 🧿 Face-Centric Scan: Focusing on {w}x{h} region.")

            if img_target is None:
                img_target = Image.open(image_path).convert("RGB")
            
            img_optimized = img_target.resize((224, 224), Image.LANCZOS)
            results = detector(img_optimized)
            
            ai_score = 0
            label = "Unknown"
            for r in results:
                lbl = r["label"].lower()
                if any(k in lbl for k in ["artificial", "fake", "ai", "generated", "synthetic"]):
                    ai_score = int(r["score"] * 100)
                    label = r["label"]
                    break
            
            return {
                "ai_probability": ai_score,
                "label": label,
                "method": "ViT (Face-Centric Boost)" if img_cv is not None else "ViT (Full Scan)",
                "model_used": True,
                "face_metrics": face_metrics
            }
        except Exception:
            pass  # Fall through to heuristic path
        finally:
            # Explicit cleanup
            if img_cv is not None: del img_cv
            gc.collect()

    # ── Heuristic fallback ────────────────────────────────────────────────────
    return _heuristic_ai_detection(image_path)


def _heuristic_ai_detection(image_path: str) -> dict:
    """
    Fallback: detects likely AI-generated images via resolution + frequency analysis.
    AI generators (Stable Diffusion, Midjourney) produce images at exact power-of-2
    resolutions and have characteristic high-frequency smoothness patterns.
    """
    score = 0
    reasons = []
    try:
        img = Image.open(image_path)
        w, h = img.size

        # Known AI output resolutions
        ai_resolutions = {(512,512),(768,768),(1024,1024),(1280,720),(1920,1080),(2048,2048)}
        if (w, h) in ai_resolutions:
            score += 30
            reasons.append(f"Resolution {w}×{h} matches known AI generator output size")

        # FFT analysis: Detect periodic grid artifacts (common in GAN/AI synthesis)
        gray = np.array(img.convert("L")).astype(np.float32)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Analyze spectral energy distribution
        log_mag = np.log(magnitude + 1)
        smoothness = np.std(log_mag)
        
        # Detect periodic spikes (spectral peaks away from center)
        # We look for outliers in the magnitude spectrum excluding the DC component
        h, w = magnitude.shape
        cy, cx = h//2, w//2
        mask = np.ones((h, w), bool)
        mask[cy-5:cy+6, cx-5:cx+6] = False # ignore center DC
        peaks = np.percentile(magnitude[mask], 99.9)
        mean_energy = np.mean(magnitude[mask])
        
        if smoothness < 3.2:
            score += 25
            reasons.append(f"Spectral smoothness detected (std={smoothness:.2f}) — typical of generative models")
        
        if peaks > mean_energy * 50:
            score += 35
            reasons.append("Periodic spectral peaks detected — structural artifacts characteristic of GAN/AI grid sampling")

    except Exception:
        pass

    return {
        "ai_probability": min(100, score),
        "label": "Likely AI-Generated" if score >= 30 else "Likely Real",
        "method": "Heuristic (resolution + FFT smoothness)",
        "model_used": False
    }


# ── ELA (kept from original) ─────────────────────────────────────────────────
def error_level_analysis(image_path, quality=90):
    original = None
    tmp_ela = None
    ela_map_path = None
    try:
        import tempfile
        original = Image.open(image_path).convert('RGB')
        tmp_ela = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        tmp_ela.close()
        original.save(tmp_ela.name, 'JPEG', quality=quality)
        compressed = Image.open(tmp_ela.name)
        diff = np.abs(np.array(original).astype(np.int16) - np.array(compressed).astype(np.int16))
        max_diff = np.max(diff) or 1
        # TIER 3: Adaptive ELA Scaling
        # If very high quality (small diff), we need a higher scale to see the signal.
        scale = 255.0 / max_diff if max_diff > 10 else 25.0 
        enhanced_diff = (diff * scale).clip(0, 255).astype(np.uint8)
        ela_out = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        ela_out.close()
        ela_map_path = ela_out.name
        Image.fromarray(enhanced_diff).save(ela_map_path)
        os.remove(tmp_ela.name)
        return {'std_diff': float(np.std(diff)), 'avg_diff': float(np.mean(diff)), 'ela_map_path': ela_map_path}
    except Exception as e:
        return {'error': str(e)}
    finally:
        if original:
            original.close()


def blur_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {'variance': 0, 'is_blurry': False}
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return {'variance': float(variance), 'is_blurry': variance < 100.0}


def analyze_image(image_path: str, low_resource: bool = False) -> dict:
    """
    Combines ELA analysis and Deep Learning (ViT) detection.
    """
    score = 0
    reasons = []

    # 1. AI-generation check
    ai_res = detect_ai_generated(image_path)
    ai_prob = int(ai_res.get("ai_probability") or 0)

    # Face-Centric and Noise Consistency Signals
    score += ai_prob
    if ai_prob >= 50:
        reasons.append(f"CRITICAL: High AI-generation signal ({ai_prob}%) [{ai_res.get('method', 'N/A')}]")
    elif ai_prob >= 20:
        reasons.append(f"AI-signature detected (Prob: {ai_prob}%)")

    # 1.1 Noise Consistency Check (Deepfake Artifacts)
    # If the face focus detected a face, compare it to the background noise
    # (Using Laplacian variance as a proxy for digital noise consistency)
    try:
        if 'face_metrics' in ai_res:
            f_var = ai_res['face_metrics']['face_var']
            bg_var = ai_res['face_metrics']['bg_var']
            ratio = f_var / (bg_var + 1e-9)
            if ratio < 0.4 or ratio > 2.5:
                score += 35
                reasons.append(f"Anomaly: Noise inconsistency between Face and Background (Ratio: {ratio:.1f}) — typical of deepfake compositing.")
    except Exception:
        pass

    # 2. ELA
    ela_res = error_level_analysis(image_path)
    ela_map = None
    ela_std = 0.0
    if isinstance(ela_res, dict) and 'std_diff' in ela_res:
        ela_std = float(ela_res['std_diff'])
        ela_map = ela_res.get('ela_map_path')
        if ela_std > 15.0:
            score += 35
            reasons.append(f"High ELA variance (Std Dev: {ela_std:.2f}) — possible splicing/editing")
        elif ela_std > 8.0:
            score += 15
            reasons.append(f"Moderate ELA variance (Std Dev: {ela_std:.2f}) — minor editing suspected")

    # 3. Blur
    blur_res = blur_detection(image_path)
    blur_variance = float(blur_res.get('variance') or 0)
    if blur_res.get('is_blurry'):
        score += 15
        reasons.append(f"Image is blurry (Laplacian variance {blur_variance:.2f}) — may hide forgery artifacts")

    final_score = min(100, score)
    return {
        'score': final_score,
        'risk_level': 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low',
        'reasons': reasons,
        'ela_map': ela_map,
        'ai_detection': ai_res,
        'metrics': {
            'ai_probability': ai_prob,
            'ela_std_dev': ela_std,
            'laplacian_variance': blur_variance,
            'model_used': bool(ai_res.get('model_used'))
        }
    }
