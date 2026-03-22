"""
TrueSight — Definitive Master Image Forensics v3.0
modules/image.py
"""

import cv2
import numpy as np
from scipy.stats import kurtosis
from scipy.ndimage import gaussian_filter
from PIL import Image as PILImage
from typing import Optional
import warnings
from config import CFG

try:
    from transformers import pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────
#  Internal Helper / Local Fusion
# ─────────────────────────────────────────────────────────────────
def _fuse_results(detectors: dict) -> dict:
    """Module-level fusion following v3.0 logic."""
    all_reasons = []
    for d in detectors.values():
        all_reasons.extend(d.get("reasons", []))

    strong = [d for d in detectors.values()
              if d.get("is_strong") and d.get("confidence", 0) > 0.5]

    if strong:
        base = max(d["score"] for d in strong)
        boost = min(
            sum(d["score"] * d.get("confidence", 0.5) * CFG.FUSION_BOOST_MULTIPLIER
                for d in detectors.values()
                if not d.get("is_strong") and d.get("score", 0) > 10),
            CFG.FUSION_BOOST_MAX
        )
        final_score = min(base + boost, 100.0)
    else:
        weighted = [
            (d["score"] * d.get("confidence", 0.5), d.get("confidence", 0.5))
            for d in detectors.values()
            if d.get("confidence", 0) > 0.3 and d.get("score", 0) > 0
        ]
        if not weighted:
            final_score = 10.0
        else:
            scores, confs = zip(*weighted)
            final_score = sum(scores) / sum(confs)

    overall_conf = float(np.mean([d.get("confidence", 0.5) for d in detectors.values()]))
    
    return {
        "score": round(final_score, 1),
        "confidence": round(overall_conf, 2),
        "is_strong": len(strong) > 0,
        "reasons": all_reasons,
        "sub_scores": {k: {"score": v["score"], "confidence": v["confidence"]} for k, v in detectors.items()}
    }

def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _radial_profile(magnitude: np.ndarray) -> np.ndarray:
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(magnitude.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    r_max = min(cx, cy)
    profile = np.array([magnitude[r == i].mean() if np.any(r == i) else 0 for i in range(1, r_max)])
    return profile

# ─────────────────────────────────────────────────────────────────
#  Detectors
# ─────────────────────────────────────────────────────────────────

def detect_ela(img_bgr: np.ndarray, image_path: str, source: str = "image") -> dict:
    try:
        buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, CFG.ELA_QUALITY])[1]
        recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img_bgr.astype(np.float32), recompressed.astype(np.float32))
        ela_map = diff * 20.0
        ela_mean = float(np.mean(ela_map))
        ela_std = float(np.std(ela_map))

        mean_thr = (
            CFG.ELA_MEAN_THRESHOLD_VIDEO
            if source == "video"
            else CFG.ELA_MEAN_THRESHOLD
        )
        std_thr = (
            CFG.ELA_STD_THRESHOLD_VIDEO
            if source == "video"
            else CFG.ELA_STD_THRESHOLD
        )

        risk = 0
        reasons = []
        if ela_mean > mean_thr:
            risk += 35
            reasons.append(
                f"[ELA] High re-compression error (mean={ela_mean:.2f}, thr={mean_thr}). Potential manipulation."
            )
        if ela_std > std_thr:
            risk += 17
            reasons.append(
                f"[ELA] Heterogeneous compression distribution (std={ela_std:.2f}, thr={std_thr})."
            )

        conf = 0.85
        is_strong = ela_mean > mean_thr and conf >= 0.85
        return {
            "score": min(risk, 100),
            "confidence": conf,
            "is_strong": is_strong,
            "reasons": reasons,
        }
    except Exception:
        return {"score": 0, "confidence": 0.1, "is_strong": False, "reasons": []}

def detect_srm_residuals(img_bgr: np.ndarray, source: str = "image") -> dict:
    gray = _to_gray(img_bgr).astype(np.float32)
    kernels = {
        "h": np.array([[-1, 2, -1]], dtype=np.float32),
        "d": np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]], dtype=np.float32),
        "s": np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0
    }
    
    clean_std_thresh = CFG.SRM_CLEAN_STD_IMAGE if source == "image" else CFG.SRM_CLEAN_STD_VIDEO
    kurt_thresh = CFG.SRM_KURTOSIS_IMAGE if source == "image" else CFG.SRM_KURTOSIS_VIDEO
    
    risk_accum = 0
    reasons = []
    agree_count = 0
    
    for name, k in kernels.items():
        res = cv2.filter2D(gray, -1, k)
        std = float(np.std(res))
        kurt = float(kurtosis(res.flatten(), fisher=True))
        
        if std < clean_std_thresh or kurt > kurt_thresh:
            agree_count += 1
            risk_accum += 15
            reasons.append(f"[SRM:{name}] Residual anomaly (std={std:.2f}, kurt={kurt:.1f}).")

    if source == "video" and agree_count < 2:
        risk_accum //= 2  # Single-filter match in video is likely codec noise

    if source == "video":
        srm_strong = agree_count >= 2 and risk_accum >= 45
    else:
        srm_strong = risk_accum >= 30

    return {
        "score": min(risk_accum, 100),
        "confidence": 0.85 if source == "image" else 0.45,
        "is_strong": srm_strong,
        "reasons": reasons,
    }

def detect_spectral_slope(img_bgr: np.ndarray, source: str = "image") -> dict:
    if source == "video":
        return {"score": 0, "confidence": 0.25, "is_strong": False, "reasons": []}

    if img_bgr.shape[0] < CFG.SPECTRAL_MIN_IMAGE_SIZE or img_bgr.shape[1] < CFG.SPECTRAL_MIN_IMAGE_SIZE:
        return {
            "score": 0,
            "confidence": 0.1,
            "is_strong": False,
            "reasons": ["[SPECTRAL] Image too small for reliable spectral analysis."],
        }

    gray = _to_gray(img_bgr).astype(np.float32)
    grad = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0)**2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1)**2)
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(grad)))
    profile = _radial_profile(magnitude)
    
    n = len(profile)
    start, end = max(int(n * CFG.SPECTRAL_SLOPE_FIT_START), 5), int(n * CFG.SPECTRAL_SLOPE_FIT_END)
    if end <= start: return {"score": 0, "confidence": 0.1, "is_strong": False, "reasons": []}
    
    log_r = np.log(np.arange(1, end - start + 1))
    log_p = np.log(profile[start:end] + 1e-10)
    slope, _ = np.polyfit(log_r, log_p, 1)
    
    deviation = abs(slope - CFG.SPECTRAL_SLOPE_CENTER)
    if deviation > CFG.SPECTRAL_SLOPE_TOLERANCE:
        return {"score": 25, "confidence": 0.80, "is_strong": False,
                "reasons": [f"[SPECTRAL] 1/f² law violation — slope={slope:.3f}, dev={deviation:.3f}."]}
    return {"score": 0, "confidence": 0.80, "is_strong": False, "reasons": []}

def detect_chromatic_aberration(img_bgr: np.ndarray, source: str = "image") -> dict:
    if source == "video":
        return {"score": 0, "confidence": 0.25, "is_strong": False, "reasons": []}

    b, g, r = cv2.split(img_bgr.astype(np.float32))
    f_r, f_b = np.fft.fft2(r), np.fft.fft2(b)
    cross = (f_r * np.conj(f_b)) / (np.abs(f_r * np.conj(f_b)) + 1e-10)
    corr = np.abs(np.fft.ifft2(cross))
    peak = np.unravel_index(corr.argmax(), corr.shape)
    h, w = corr.shape
    dy = peak[0] if peak[0] < h//2 else peak[0] - h
    dx = peak[1] if peak[1] < w//2 else peak[1] - w
    shift_px = float(np.sqrt(dx**2 + dy**2))

    if shift_px < CFG.CHROMA_ALIGN_SYNTHETIC_MAX:
        return {"score": 20, "confidence": 0.75, "is_strong": False,
                "reasons": [f"[CHROMA] suspiciously perfect R/B alignment — shift={shift_px:.3f} px."]}
    if shift_px > CFG.CHROMA_WARP_SUSPICIOUS:
        return {"score": 10, "confidence": 0.60, "is_strong": False,
                "reasons": [f"[CHROMA] Extreme shift={shift_px:.2f} px (heavy warping)."]}
    return {
        "score": 0,
        "confidence": 0.75,
        "is_strong": False,
        "reasons": [],
    }


def _dct_block_boundary_ratio(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    h, w = g.shape
    if h < 16 or w < 16:
        return 0.0
    on_x = float(np.mean(gx[:, 7::8])) if w > 8 else 0.0
    off_x = float(np.mean(gx[:, 3::8])) if w > 8 else 0.0
    on_y = float(np.mean(gy[7::8, :])) if h > 8 else 0.0
    off_y = float(np.mean(gy[3::8, :])) if h > 8 else 0.0
    boundary = (on_x + on_y) / 2.0
    interior = (off_x + off_y) / 2.0 + 1e-9
    return boundary / interior


def detect_noise_floor(img_bgr: np.ndarray) -> dict:
    gray = _to_gray(img_bgr).astype(np.float32)
    noise = gray - cv2.GaussianBlur(gray, (5, 5), 0)
    noise_std = float(np.std(noise))
    risk = 20 if noise_std < CFG.NOISE_FLOOR_SYNTHETIC_MAX else 0
    reasons = [f"[NOISE] Sterile AI noise floor (std={noise_std:.3f})."] if risk else []
    
    h, w = noise.shape
    blocks = [noise[y:y+32, x:x+32].std() for y in range(0, h-32, 32) for x in range(0, w-32, 32)]
    if blocks and (np.std(blocks) / (np.mean(blocks) + 1e-9)) > CFG.NOISE_SPATIAL_CV_MAX:
        risk += 10
        reasons.append(f"[NOISE] Spatially non-uniform noise grain.")
        
    return {"score": min(risk, 100), "confidence": 0.80, "is_strong": False, "reasons": reasons}

def detect_dct_grid(img_bgr: np.ndarray, source: str = "image") -> dict:
    if source == "video":
        return {"score": 0, "confidence": 0.25, "is_strong": False, "reasons": []}

    gray = _to_gray(img_bgr).astype(np.float32)
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    magnitude[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0
    ratio = float(magnitude.max() / (magnitude.mean() + 1e-10))

    risk = 40 if ratio > CFG.DCT_GRID_PEAK_RATIO else 0
    reasons = (
        [f"[DCT-GRID] FFT structural grid detected (ratio={ratio:.1f}). GAN/Upscaler signature."]
        if risk
        else []
    )

    br = _dct_block_boundary_ratio(gray)
    if br > CFG.DCT_BOUNDARY_MISMATCH:
        risk = min(100, risk + 20)
        reasons.append(
            f"[DCT-GRID] 8×8 block boundary mismatch ratio={br:.3f} (>{CFG.DCT_BOUNDARY_MISMATCH})."
        )

    is_strong = ratio > CFG.DCT_GRID_PEAK_RATIO
    return {
        "score": risk,
        "confidence": 0.90,
        "is_strong": is_strong,
        "reasons": reasons,
    }

def detect_copy_move(img_bgr: np.ndarray, source: str = "image") -> dict:
    gray = _to_gray(img_bgr)
    sift = cv2.SIFT_create(nfeatures=500)
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) < 20:
        return {"score": 0, "confidence": 0.2, "is_strong": False, "reasons": []}

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des, des, k=3)
    suspicious = 0
    for mset in matches:
        if len(mset) < 3:
            continue
        non_self = [m for m in mset if m.queryIdx != m.trainIdx]
        if len(non_self) < 2:
            continue
        m_a, m_b = non_self[0], non_self[1]
        if m_a.distance >= 0.6 * m_b.distance:
            continue
        p1 = kp[m_a.queryIdx].pt
        p2 = kp[m_a.trainIdx].pt
        if np.hypot(p1[0] - p2[0], p1[1] - p2[1]) > 20:
            suspicious += 1

    need = (
        CFG.COPY_MOVE_MIN_MATCHES_VIDEO
        if source == "video"
        else CFG.COPY_MOVE_MIN_MATCHES
    )
    risk = 35 if suspicious >= need else 0
    reasons = (
        [f"[COPY-MOVE] {suspicious} suspicious self-matches (copy-move)."]
        if risk
        else []
    )
    conf = 0.70
    is_strong = suspicious >= need and conf >= 0.70
    return {"score": risk, "confidence": conf, "is_strong": is_strong, "reasons": reasons}

def detect_ai_resolution(img_bgr: np.ndarray) -> dict:
    h, w = img_bgr.shape[:2]
    if (w, h) in CFG.AI_RESOLUTIONS:
        return {"score": 15, "confidence": 0.60, "is_strong": False, "reasons": [f"[RESOLUTION] Exact AI output size {w}x{h} detected."]}
    return {"score": 0, "confidence": 0.90, "is_strong": False, "reasons": []}

def detect_metadata_anomalies(image_path: str) -> dict:
    risk, reasons = 0, []
    png_ai_keys = (
        "parameters",
        "prompt",
        "cfg scale",
        "sampler",
        "seed",
        "model",
        "steps",
    )
    try:
        img = PILImage.open(image_path)
        info = {str(k).lower(): str(v).lower() for k, v in (img.info or {}).items()}
        text = " ".join(info.values())
        ai_hit = any(tag in text for tag in CFG.AI_SOFTWARE_TAGS)
        if ai_hit:
            risk += 30
            reasons.append("[METADATA] AI software signature found in tags.")
        for pk in png_ai_keys:
            if any(pk in k or pk in v for k, v in info.items()):
                risk += 25
                reasons.append(f"[METADATA] Generative-AI PNG/text chunk hint ({pk}).")
                ai_hit = True
                break
        try:
            with open(image_path, "rb") as f:
                head = f.read(65536)
            if b"c2pa" in head or b"cai\x00" in head:
                risk += 15
                reasons.append("[METADATA] C2PA / content-credentials marker in file header.")
        except OSError:
            pass

        if image_path.lower().endswith((".jpg", ".jpeg")):
            exif = img._getexif()
            if not exif:
                risk += 10
                reasons.append("[METADATA] Missing EXIF header on JPEG.")
        conf = 0.90
        is_strong = ai_hit and conf >= 0.90
        return {"score": risk, "confidence": conf, "is_strong": is_strong, "reasons": reasons}
    except Exception:
        return {"score": 0, "confidence": 0.1, "is_strong": False, "reasons": []}

def _load_detector():
    """Optional ViT / transformer pipeline; Streamlit cache calls this. Heuristics work without it."""
    return None


# ─────────────────────────────────────────────────────────────────
#  Master Entry Point
# ─────────────────────────────────────────────────────────────────
def analyze_image(image_path: str, source: str = "image") -> dict:
    img = cv2.imread(image_path)
    if img is None or img.shape[0] < 32 or img.shape[1] < 32:
        return {"score": 10, "confidence": 0.1, "is_strong": False, "reasons": ["Unreadable image."], "sub_scores": {}}
    
    detectors = {
        "ela":       detect_ela(img, image_path, source),
        "srm":       detect_srm_residuals(img, source),
        "spectral":  detect_spectral_slope(img, source),
        "chroma":    detect_chromatic_aberration(img, source),
        "noise":     detect_noise_floor(img),
        "dct_grid":  detect_dct_grid(img, source),
        "copy_move": detect_copy_move(img, source),
        "ai_res":    detect_ai_resolution(img),
        "metadata":  detect_metadata_anomalies(image_path),
    }
    return _fuse_results(detectors)
