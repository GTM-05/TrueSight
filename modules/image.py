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


def _srm_lite_residual(img_bgr: np.ndarray) -> dict:
    """
    Lightweight 3-filter SRM (Steganalysis Rich Model) residual.
    Detects manipulation noise patterns invisible to the human eye.
    """
    if img_bgr is None: return {'score': 0, 'confidence': 0, 'is_strong': False, 'reasons': []}
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # SRM filters
    k1 = np.array([[-1, 2, -1]], dtype=np.float32)
    k2 = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]], dtype=np.float32)
    k3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0

    residuals = [cv2.filter2D(gray, -1, k) for k in [k1, k2, k3]]
    
    risk = 0
    reasons = []
    
    for i, res in enumerate(residuals):
        res_std = float(np.std(res))
        res_kurtosis = float(np.mean((res - np.mean(res))**4) / (res_std**4 + 1e-9))
        
        if res_std < 8.0:
            risk += 15
            reasons.append(f"SRM residual {i+1}: clean noise floor (std={res_std:.2f})")
        if res_kurtosis > 10.0:
            risk += 10
            reasons.append(f"SRM residual {i+1}: manipulation artifacts (kurtosis={res_kurtosis:.1f})")
            
    return {
        'score': min(risk, 45),
        'confidence': 0.8,
        'is_strong': risk >= 30,
        'reasons': reasons
    }


def detect_ai_generated(image_path: str, low_resource: bool = False, face_focus: bool = True) -> dict:
    """
    Uses a pre-trained ViT model to score the probability that an image
    was AI-generated. Returns a score 0–100 and a label.
    """
    detector = _load_detector()
    face_cascade = _get_face_detector()
    
    ai_score = 0
    label = "Unknown"
    confidence = 0.5
    is_strong = False
    texture_bonus = 0
    face_metrics = {}

    # ── ML path ──────────────────────────────────────────────────────────────
    if not low_resource and detector is not None:
        img_cv = cv2.imread(image_path)
        img_target = None
        try:
            if img_cv is not None and face_focus and face_cascade is not None:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                full_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                faces = face_cascade.detectMultiScale(gray, 1.05, 3)
                
                if len(faces) > 0:
                    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                    face_crop = img_cv[max(0,y-10):y+h+10, max(0,x-10):x+w+10]
                    face_var = cv2.Laplacian(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    face_metrics = {'face_var': float(face_var), 'bg_var': float(full_var)}
                    img_target = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    confidence = 0.85

                    # Micro-texture check
                    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    denoised = cv2.fastNlMeansDenoising(gray_crop, h=10)
                    noise_map = cv2.absdiff(gray_crop, denoised)
                    noise_std = float(np.std(noise_map))
                    if noise_std < 0.85:
                        texture_bonus = 35
                        face_metrics['texture_anomaly'] = 1.0

            if img_target is None:
                img_target = Image.open(image_path).convert("RGB")
                confidence = 0.6
            
            img_optimized = img_target.resize((224, 224), Image.LANCZOS)
            results = detector(img_optimized)
            
            for r in results:
                lbl = r["label"].lower()
                if any(k in lbl for k in ["artificial", "fake", "ai", "generated", "synthetic"]):
                    ai_score = int(r["score"] * 100) + texture_bonus
                    label = r["label"]
                    is_strong = (r["score"] > 0.45)
                    break
            
            return {
                "ai_probability": min(100, ai_score),
                "label": label,
                "confidence": confidence,
                "is_strong": is_strong,
                "method": "ViT",
                "face_metrics": face_metrics
            }
        except Exception:
            pass
        finally:
            if 'img_cv' in locals() and img_cv is not None: del img_cv
            gc.collect()

    return _heuristic_ai_detection(image_path)


def _chromatic_aberration_score(img_cv: np.ndarray) -> float:
    """
    Real cameras have slight R/B channel misalignment (chromatic aberration).
    AI generators often produce perfectly aligned channels.
    A suspiciously LOW score (< 2.0) indicates synthetic generation.
    """
    if img_cv is None: return 0.0
    b, g, r = cv2.split(img_cv)
    # Compare Red and Blue channels for displacement
    rb_diff = cv2.absdiff(r, b)
    return float(np.mean(rb_diff))

def _noise_floor_analysis(img_cv: np.ndarray) -> float:
    """
    Real sensors have ISO noise. AI images often have suspiciously clean/uniform noise floors.
    Too low std (< 0.5) in the residual indicates synthetic origin.
    """
    if img_cv is None: return 0.0
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Extract noise residual using a high-pass filter or thresholding zero
    _, residual = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO)
    return float(np.std(residual))

def _spectral_analysis(image_path: str) -> dict:
    """
    Analyzes the 2D Power Spectrum of an image to detect GAN/AI grid artifacts.
    Returns a score 0-60 and reasons.
    """
    score = 0
    reasons = []
    try:
        img = Image.open(image_path)
        # FFT analysis: Detect periodic grid artifacts (common in GAN/AI synthesis)
        gray = np.array(img.convert("L")).astype(np.float32)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Analyze spectral energy distribution
        log_mag = np.log(magnitude + 1)
        smoothness = np.std(log_mag)
        
        # Detect periodic spikes (spectral peaks away from center)
        h, w = magnitude.shape
        cy, cx = h//2, w//2
        mask = np.ones((h, w), bool)
        mask[cy-5:cy+6, cx-5:cx+6] = False # ignore center DC
        
        # Avoid division by zero
        if np.any(mask):
            peaks = np.percentile(magnitude[mask], 99.9)
            mean_energy = np.mean(magnitude[mask])
        else:
            peaks, mean_energy = 0, 1
        
        # 1. Spectral Smoothness (Existing)
        if smoothness < 2.35:
            score += 20
            reasons.append(f"Spectral smoothness detected (std={smoothness:.2f})")
        
        # 2. Grid detection (Existing)
        structural_artifact = False
        peak_ratio = float(peaks / (mean_energy + 1e-9))
        if peak_ratio > 120:
            score += 40 
            reasons.append(f"Periodic spectral peaks detected (Ratio: {peak_ratio:.1f}) — structural AI artifacts (Grid)")
            structural_artifact = True
            
        # 3. NEW: Radial Power Spectrum Slope
        # Real images follow a 1/f^2 law (Slope ~ -2.2). AI deviates.
        h, w = magnitude.shape
        y, x = np.indices((h, w))
        r = np.sqrt((x - w//2)**2 + (y - h//2)**2).astype(np.int32)
        tbin = np.bincount(r.ravel(), magnitude.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-9)
        
        # Fit log-log slope for mid-frequencies
        indices = np.arange(5, min(h, w)//2 - 5)
        if len(indices) > 5:
            log_r = np.log(indices)
            log_p = np.log(radial_profile[indices] + 1e-9)
            slope, _ = np.polyfit(log_r, log_p, 1)
            # Normal: ~ -2.2. AI: deviates heavily
            slope_anomaly = abs(slope - (-2.2)) > 0.5
            if slope_anomaly:
                score += 25
                reasons.append(f"Radial spectral slope anomaly (Slope: {slope:.2f}) — unnatural frequency distribution.")
        else:
            slope = -2.2
            slope_anomaly = False
            
    except Exception:
        pass
    
    return {
        'score': score, 
        'reasons': reasons, 
        'smoothness': float(smoothness) if 'smoothness' in locals() else 0.0,
        'structural_artifact': structural_artifact,
        'peak_ratio': peak_ratio if 'peak_ratio' in locals() else 0.0,
        'spectral_slope': float(slope) if 'slope' in locals() else -2.2,
        'slope_anomaly': slope_anomaly if 'slope_anomaly' in locals() else False
    }


def _heuristic_ai_detection(image_path: str) -> dict:
    """
    Fallback: detects likely AI-generated images via resolution + frequency analysis.
    """
    score = 0
    reasons = []
    try:
        img = Image.open(image_path)
        w, h = img.size
        # Suspicious square resolutions typical of generative models
        ai_resolutions = {(512,512),(768,768),(1024,1024),(2048,2048)}
        if (w, h) in ai_resolutions:
            score += 20
            reasons.append(f"Fixed square resolution {w}×{h} detected — suspicious AI footprint")
        
        spec_res = _spectral_analysis(image_path)
        score += spec_res['score']
        reasons.extend(spec_res['reasons'])
        
        # NEW: Modern Heuristics (Chromatic & Noise)
        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            chrom_score = _chromatic_aberration_score(img_cv)
            if chrom_score < 2.5: # Suspiciously low alignment
                score += 25
                reasons.append(f"Chromatic alignment anomaly (Score: {chrom_score:.2f}) — typical of GAN/Diffusion synthesis.")
            
            noise_std = _noise_floor_analysis(img_cv)
            if noise_std < 0.65: # Suspiciously clean
                score += 20
                reasons.append(f"Synthetic noise floor detected (std={noise_std:.2f}) — lacking natural sensor grain.")
    except Exception:
        pass

    return {
        "ai_probability": min(100, score),
        "label": "Likely AI-Generated" if score >= 30 else "Likely Real",
        "confidence": 0.5 if score < 20 else 0.7,
        "is_strong": score >= 45,
        "method": "Heuristic (resolution + spectral)",
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
    Combines ELA, SRM-Lite, Spectral, and Deep Learning (ViT) detection.
    """
    score = 0
    reasons = []
    
    img_cv = cv2.imread(image_path)

    # 1. AI-generation check (ViT / Layout)
    ai_res = detect_ai_generated(image_path)
    ai_prob = int(ai_res.get("ai_probability") or 0)
    score += ai_prob
    if ai_prob >= 50:
        reasons.append(f"CRITICAL: High AI-generation signal ({ai_prob}%)")
    elif ai_prob >= 20:
        reasons.append(f"AI-signature detected (Prob: {ai_prob}%)")

    # 2. SRM-Lite (Steganalysis)
    srm_res = _srm_lite_residual(img_cv)
    score += srm_res['score']
    reasons.extend(srm_res['reasons'])

    # 3. Spectral Analysis
    spec_res = _spectral_analysis(image_path)
    # Heuristic detection already included spectral, but we ensure its metrics are exposed

    # 4. Noise Consistency Check
    try:
        if 'face_metrics' in ai_res:
            f_var = ai_res['face_metrics']['face_var']
            bg_var = ai_res['face_metrics']['bg_var']
            ratio = f_var / (bg_var + 1e-9)
            if ratio < 0.4 or ratio > 2.5:
                score += 35
                reasons.append(f"Anomaly: Noise inconsistency Fac/BG (Ratio: {ratio:.1f})")
    except Exception:
        pass

    # 5. ELA
    ela_res = error_level_analysis(image_path)
    ela_map = None
    ela_std = 0.0
    if isinstance(ela_res, dict) and 'std_diff' in ela_res:
        ela_std = float(ela_res['std_diff'])
        ela_map = ela_res.get('ela_map_path')
        if ela_std > 15.0:
            score += 35
            reasons.append(f"High ELA variance (Std Dev: {ela_std:.2f})")
        elif ela_std > 8.0:
            score += 15

    # 6. Blur
    blur_res = blur_detection(image_path)
    blur_variance = float(blur_res.get('variance') or 0)
    if blur_res.get('is_blurry') and score < 50: # Only add if we don't already have strong signal
        score += 15

    final_score = min(100, score)
    
    # Confidence calculation: Weighted average of sub-confidence
    confidence = float(np.mean([
        ai_res.get('confidence', 0.5),
        srm_res.get('confidence', 0.8),
        0.7 # Baseline for ELA/Spectral
    ]))
    
    is_strong = (ai_prob > 40 or srm_res['is_strong'] or ela_std > 12)

    return {
        'score': final_score,
        'risk_level': 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low',
        'reasons': list(set(reasons)),
        'ela_map': ela_map,
        'ai_detection': ai_res,
        'metrics': {
            'ai_probability': ai_prob,
            'ela_std_dev': ela_std,
            'laplacian_variance': blur_variance,
            'srm_score': srm_res['score'],
            'structural_artifact': spec_res.get('structural_artifact', False),
            'spectral_slope': spec_res.get('spectral_slope', -2.2)
        },
        'confidence': confidence,
        'is_strong': is_strong
    }
