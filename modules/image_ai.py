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

# ── HuggingFace AI-image detector (loaded lazily on first call) ──────────────
_ai_detector = None
_detector_ready = False

def _load_detector():
    global _ai_detector, _detector_ready
    if _detector_ready:
        return _ai_detector
    try:
        from transformers import pipeline
        # umm-maybe/AI-image-detector: ViT fine-tuned to detect AI-generated images
        _ai_detector = pipeline(
            "image-classification",
            model="umm-maybe/AI-image-detector",
            device=-1  # CPU only
        )
        _detector_ready = True
    except Exception:
        _ai_detector = None
        _detector_ready = True  # Don't retry, use fallback
    return _ai_detector


def detect_ai_generated(image_path: str) -> dict:
    """
    Uses a pre-trained ViT model to score the probability that an image
    was AI-generated. Returns a score 0–100 and a label.
    Falls back to heuristic indicators if model is unavailable.
    """
    detector = _load_detector()

    # ── ML path ──────────────────────────────────────────────────────────────
    if detector is not None:
        try:
            img = Image.open(image_path).convert("RGB")
            results = detector(img)
            # Results: [{"label": "artificial", "score": 0.93}, {"label": "human", ...}]
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
                "method": "ViT (umm-maybe/AI-image-detector)",
                "model_used": True
            }
        except Exception as e:
            pass  # Fall through to heuristic path

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

        # FFT smoothness: AI images tend to have very smooth frequency spectrum
        gray = np.array(img.convert("L")).astype(np.float32)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)
        smoothness = np.std(magnitude)
        if smoothness < 3.5:
            score += 25
            reasons.append(f"Unusually smooth frequency spectrum (std={smoothness:.2f}) — characteristic of AI synthesis")

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
    try:
        original = Image.open(image_path).convert('RGB')
        temp_filename = "temp_ela_ai.jpg"
        original.save(temp_filename, 'JPEG', quality=quality)
        compressed = Image.open(temp_filename)
        diff = np.abs(np.array(original).astype(np.int16) - np.array(compressed).astype(np.int16))
        max_diff = np.max(diff) or 1
        scale = 255.0 / max_diff
        enhanced_diff = (diff * scale).astype(np.uint8)
        ela_map_path = "ela_result_ai.jpg"
        Image.fromarray(enhanced_diff).save(ela_map_path)
        os.remove(temp_filename)
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


def analyze_image_ai(image_path: str) -> dict:
    """
    Full enhanced image analysis:
      1. AI-generation detection (ML model or heuristic fallback)
      2. ELA manipulation detection (heuristic)
      3. Blur detection (heuristic)
    Returns a merged score, reasons, and all evidence.
    """
    score = 0
    reasons = []

    # 1. AI-generation check
    ai_res = detect_ai_generated(image_path)
    ai_prob = int(ai_res.get("ai_probability") or 0)
    if ai_prob >= 70:
        score += 50
        reasons.append(f"High AI-generation probability: {ai_prob}% [{ai_res.get('method', 'N/A')}]")
    elif ai_prob >= 40:
        score += 25
        reasons.append(f"Moderate AI-generation probability: {ai_prob}% [{ai_res.get('method', 'N/A')}]")

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
