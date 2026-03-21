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
        from transformers import pipeline
        import torch
        # jacoballessio/ai-image-detect-distilled: 11.8M params, 58MB.
        # Much faster and lighter for 8GB RAM than base ViT.
        device = 0 if torch.cuda.is_available() else -1
        _ai_detector = pipeline(
            "image-classification",
            model="jacoballessio/ai-image-detect-distilled",
            device=device
        )
        _detector_ready = True
    except Exception:
        _ai_detector = None
        _detector_ready = True  # Don't retry, use fallback
    return _ai_detector


def detect_ai_generated(image_path: str, low_resource: bool = False) -> dict:
    """
    Uses a pre-trained ViT model to score the probability that an image
    was AI-generated. Returns a score 0–100 and a label.
    Falls back to heuristic indicators if model is unavailable.
    """
    detector = _load_detector()

    # ── ML path ──────────────────────────────────────────────────────────────
    # Skip ML path in Low Resource Mode (8GB RAM optimization)
    if not low_resource and detector is not None:
        img = None
        img_optimized = None
        try:
            img = Image.open(image_path).convert("RGB")
            # Optimize: Resize to 224x224 before passing to ViT to save RAM/CPU
            img_optimized = img.resize((224, 224), Image.LANCZOS)
            results = detector(img_optimized)
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
        except Exception:
            pass  # Fall through to heuristic path
        finally:
            # Explicit cleanup to recover RAM on 8GB systems
            if img is not None: del img
            if img_optimized is not None: del img_optimized
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
        scale = 255.0 / max_diff
        enhanced_diff = (diff * scale).astype(np.uint8)
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
    
    # If AI model is highly confident, it should carry a massive weight
    if ai_prob >= 80:
        score += 80
        reasons.append(f"CRITICAL: High AI-generation probability ({ai_prob}%) [{ai_res.get('method', 'N/A')}]")
    elif ai_prob >= 50:
        score += 55
        reasons.append(f"Strong AI-generation signal ({ai_prob}%) [{ai_res.get('method', 'N/A')}]")
    elif ai_prob >= 35:
        score += 35
        reasons.append(f"Moderate AI-generation artifacts (Prob: {ai_prob}%)")
    elif ai_prob >= 20:
        score += 15
        reasons.append(f"Weak AI-generation signal (Prob: {ai_prob}%)")

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
