"""
TrueSight — Strong Image Forensics Module
modules/image.py

Multi-layer image artifact detection: ELA, SRM-lite, spectral slope,
chromatic alignment, noise floor, DCT block grid, copy-move, metadata.
Pure OpenCV / NumPy / SciPy — no heavy deep learning required.
"""

import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.ndimage import uniform_filter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import struct
import io

try:
    from PIL import Image as PILImage
    import piexif
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────
@dataclass
class ImageForensicConfig:
    # ELA
    ELA_QUALITY: int = 85
    ELA_AMPLIFY: int = 20
    ELA_THRESHOLD_MEAN: float = 12.0        # > 12 = suspicious ELA mean
    ELA_THRESHOLD_STD: float = 18.0         # > 18 = suspicious ELA std

    # SRM Residual
    SRM_CLEAN_STD_MAX: float = 7.5          # Unnaturally clean residual
    SRM_KURTOSIS_HIGH: float = 9.0          # Heavy tails = manipulation

    # Spectral Slope (1/f² law)
    SPECTRAL_SLOPE_REAL_MIN: float = -2.6
    SPECTRAL_SLOPE_REAL_MAX: float = -1.8
    SPECTRAL_SLOPE_TOLERANCE: float = 0.45

    # Chromatic Alignment
    CHROMA_ALIGN_REAL_MIN: float = 1.2      # Real cameras: R/B shift > 1.2 px
    CHROMA_ALIGN_SYNTHETIC_MAX: float = 0.6 # AI: suspiciously perfect alignment

    # Noise Floor (sensor grain)
    NOISE_FLOOR_MIN_REAL: float = 2.5       # Real sensors always have grain
    NOISE_FLOOR_SYNTHETIC_MAX: float = 1.4  # AI: unnaturally clean

    # DCT Block Grid
    DCT_GRID_PEAK_RATIO: float = 110.0      # > 110 = structural grid detected
    DCT_MISMATCH_THRESHOLD: float = 0.15    # Block boundary mismatch ratio

    # Copy-Move (keypoint matching)
    COPY_MOVE_MIN_MATCHES: int = 12

    # AI Resolution Fingerprints (common generator output sizes)
    AI_RESOLUTIONS: set = field(default_factory=lambda: {
        (512, 512), (768, 768), (1024, 1024), (1024, 576), (576, 1024),
        (1280, 720), (832, 1216), (1216, 832), (1344, 768), (768, 1344),
        (896, 1152), (1152, 896), (2048, 2048), (640, 480),
    })

    # Metadata
    AI_SOFTWARE_TAGS: tuple = (
        "stable diffusion", "midjourney", "dall-e", "firefly", "imagen",
        "nightcafe", "dreamstudio", "novelai", "comfyui", "a1111",
        "automatic1111", "invoke ai", "leonardo", "ideogram", "flux",
        "generativeai", "adobe firefly", "bing image creator",
    )

    # Risk weights
    WEIGHTS: dict = field(default_factory=lambda: {
        "ela":              35,
        "srm_clean":        20,
        "srm_kurtosis":     15,
        "spectral_slope":   25,
        "chroma_align":     20,
        "noise_floor":      20,
        "dct_grid":         40,
        "dct_mismatch":     20,
        "copy_move":        35,
        "ai_resolution":    15,
        "metadata_ai":      30,
        "metadata_no_exif": 10,
        "c2pa_missing":     10,
    })


CFG = ImageForensicConfig()


# ─────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────
def _load_bgr(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _to_float32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
#  Detector 1 — Error Level Analysis (ELA)
# ─────────────────────────────────────────────────────────────────
def detect_ela(img_bgr: np.ndarray, image_path: str) -> dict:
    """
    ELA: re-compress at known quality, measure error residual.
    Tampered / AI-generated regions show abnormal ELA patterns.
    """
    risk = 0
    reasons = []

    try:
        # Save temp JPEG at reference quality
        buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, CFG.ELA_QUALITY])[1]
        recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        ela = cv2.absdiff(img_bgr.astype(np.float32),
                          recompressed.astype(np.float32))
        ela_amplified = np.clip(ela * CFG.ELA_AMPLIFY, 0, 255).astype(np.uint8)
        ela_gray = cv2.cvtColor(ela_amplified, cv2.COLOR_BGR2GRAY)

        ela_mean = float(np.mean(ela_gray))
        ela_std = float(np.std(ela_gray))

        if ela_mean > CFG.ELA_THRESHOLD_MEAN:
            risk += CFG.WEIGHTS["ela"]
            reasons.append(
                f"[ELA] High error-level mean={ela_mean:.2f} "
                f"(threshold > {CFG.ELA_THRESHOLD_MEAN}). "
                f"Indicates multiple re-saves or AI post-processing artifacts "
                f"inconsistent with a single-capture JPEG."
            )
        elif ela_std > CFG.ELA_THRESHOLD_STD:
            risk += CFG.WEIGHTS["ela"] // 2
            reasons.append(
                f"[ELA] Heterogeneous ELA distribution (std={ela_std:.2f}). "
                f"Certain regions show different compression histories — "
                f"typical of AI inpainting or compositing."
            )

        confidence = 0.85
    except Exception as e:
        return {"score": 0, "confidence": 0, "reasons": [f"ELA failed: {e}"]}

    return {
        "score": min(risk, 100),
        "confidence": confidence,
        "is_strong": risk >= 35,
        "reasons": reasons,
        "debug": {"ela_mean": ela_mean, "ela_std": ela_std}
    }


# ─────────────────────────────────────────────────────────────────
#  Detector 2 — SRM-Lite Residual Analysis
# ─────────────────────────────────────────────────────────────────
def detect_srm_residuals(img_bgr: np.ndarray) -> dict:
    """
    3-filter SRM (Steganalysis Rich Model) residuals.
    AI generators produce suspiciously clean prediction errors.
    Manipulated images show high-kurtosis residuals at boundaries.
    """
    risk = 0
    reasons = []

    gray = _to_gray(img_bgr).astype(np.float32)

    kernels = {
        "horizontal": np.array([[-1, 2, -1]], dtype=np.float32),
        "diagonal":   np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]], dtype=np.float32),
        "square":     np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
                               dtype=np.float32) / 4.0,
    }

    for name, k in kernels.items():
        res = cv2.filter2D(gray, -1, k).flatten()
        res_std = float(np.std(res))
        res_kurt = float(kurtosis(res, fisher=True))

        if res_std < CFG.SRM_CLEAN_STD_MAX:
            risk += CFG.WEIGHTS["srm_clean"] // len(kernels)
            reasons.append(
                f"[SRM:{name}] Unnaturally clean prediction residual "
                f"(std={res_std:.2f} < {CFG.SRM_CLEAN_STD_MAX}). "
                f"AI pixel generators produce sterile residuals lacking real sensor noise."
            )

        if res_kurt > CFG.SRM_KURTOSIS_HIGH:
            risk += CFG.WEIGHTS["srm_kurtosis"] // len(kernels)
            reasons.append(
                f"[SRM:{name}] High-kurtosis residual (kurt={res_kurt:.2f} "
                f"> {CFG.SRM_KURTOSIS_HIGH}). Heavy-tail distribution indicates "
                f"localized pixel manipulation boundaries."
            )

    return {
        "score": min(risk, 100),
        "confidence": 0.85,
        "is_strong": risk >= 30,
        "reasons": reasons
    }


# ─────────────────────────────────────────────────────────────────
#  Detector 3 — Radial Spectral Slope (1/f² Law)
# ─────────────────────────────────────────────────────────────────
def _radial_profile(magnitude: np.ndarray) -> np.ndarray:
    """Compute rotationally-averaged power spectrum."""
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices(magnitude.shape)
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
    r_max = min(cx, cy)
    profile = np.array([magnitude[r == i].mean() if np.any(r == i) else 0
                        for i in range(1, r_max)])
    return profile


def detect_spectral_slope(img_bgr: np.ndarray) -> dict:
    """
    Real camera images follow a 1/f² power law (slope ≈ -2.0 to -2.5).
    AI images deviate — often too flat (-1.2 to -1.8) or irregular.
    """
    risk = 0
    reasons = []

    gray = _to_gray(img_bgr).astype(np.float32)
    # Gradient magnitude before FFT (suppresses DC, enhances edges)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    fft = np.fft.fft2(grad)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    profile = _radial_profile(magnitude)
    if len(profile) < 10:
        return {"score": 0, "confidence": 0.1, "reasons": ["Image too small for spectral analysis."]}

    log_r = np.log(np.arange(1, len(profile) + 1))
    log_p = np.log(profile + 1e-10)

    # Fit slope over mid-frequency range (skip DC and extreme high-freq)
    fit_start, fit_end = max(5, len(profile) // 10), len(profile) * 3 // 4
    slope, _ = np.polyfit(log_r[fit_start:fit_end], log_p[fit_start:fit_end], 1)
    slope = float(slope)

    deviation = abs(slope - (-2.2))
    if deviation > CFG.SPECTRAL_SLOPE_TOLERANCE:
        risk += CFG.WEIGHTS["spectral_slope"]
        reasons.append(
            f"[SPECTRAL] Power-law slope deviation — measured slope={slope:.3f}, "
            f"expected range [{CFG.SPECTRAL_SLOPE_REAL_MIN}, {CFG.SPECTRAL_SLOPE_REAL_MAX}]. "
            f"Real camera images follow 1/f² statistics. "
            f"AI generators (diffusion, GANs) violate this law, especially at mid-frequencies."
        )

    return {
        "score": min(risk, 100),
        "confidence": 0.80,
        "is_strong": risk >= 25,
        "reasons": reasons,
        "debug": {"spectral_slope": slope, "deviation": deviation}
    }


# ─────────────────────────────────────────────────────────────────
#  Detector 4 — Chromatic Aberration (Channel Alignment)
# ─────────────────────────────────────────────────────────────────
def detect_chromatic_aberration(img_bgr: np.ndarray) -> dict:
    """
    Real lenses introduce lateral chromatic aberration — R/B channels
    are slightly misaligned. AI generators produce perfectly aligned channels.
    """
    risk = 0
    reasons = []

    b, g, r = cv2.split(img_bgr.astype(np.float32))

    # Phase correlation: measure sub-pixel shift between R and B channels
    def phase_shift(ch1: np.ndarray, ch2: np.ndarray) -> float:
        f1 = np.fft.fft2(ch1)
        f2 = np.fft.fft2(ch2)
        cross = f1 * np.conj(f2)
        cross /= (np.abs(cross) + 1e-10)
        corr = np.abs(np.fft.ifft2(cross))
        peak = np.unravel_index(corr.argmax(), corr.shape)
        dy = peak[0] if peak[0] < corr.shape[0] // 2 else peak[0] - corr.shape[0]
        dx = peak[1] if peak[1] < corr.shape[1] // 2 else peak[1] - corr.shape[1]
        return float(np.sqrt(dx ** 2 + dy ** 2))

    shift = phase_shift(r, b)

    if shift < CFG.CHROMA_ALIGN_SYNTHETIC_MAX:
        risk += CFG.WEIGHTS["chroma_align"]
        reasons.append(
            f"[CHROMA] Suspiciously perfect R/B channel alignment — shift={shift:.3f} px "
            f"(threshold < {CFG.CHROMA_ALIGN_SYNTHETIC_MAX} px). "
            f"Real camera lenses always produce some lateral chromatic aberration. "
            f"AI image generators render channels in perfect mathematical alignment."
        )
    elif shift > 8.0:
        # Extreme shift = possible compositing / warping
        risk += CFG.WEIGHTS["chroma_align"] // 2
        reasons.append(
            f"[CHROMA] Extreme R/B channel shift={shift:.2f} px — "
            f"may indicate aggressive warping or composite manipulation."
        )

    return {
        "score": min(risk, 100),
        "confidence": 0.75,
        "is_strong": risk >= 20,
        "reasons": reasons,
        "debug": {"rb_shift_px": shift}
    }


# ─────────────────────────────────────────────────────────────────
#  Detector 5 — Noise Floor / Sensor Grain Analysis
# ─────────────────────────────────────────────────────────────────
def detect_noise_floor(img_bgr: np.ndarray) -> dict:
    """
    Real sensors produce characteristic noise (photon shot + read noise).
    AI images have unnaturally clean noise floors — sterile digital smoothness.
    """
    risk = 0
    reasons = []

    gray = _to_gray(img_bgr).astype(np.float32)

    # High-pass filter to isolate noise from signal
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blurred

    noise_std = float(np.std(noise))
    noise_kurt = float(kurtosis(noise.flatten(), fisher=True))

    if noise_std < CFG.NOISE_FLOOR_SYNTHETIC_MAX:
        risk += CFG.WEIGHTS["noise_floor"]
        reasons.append(
            f"[NOISE] Sterile noise floor — std={noise_std:.3f} "
            f"(threshold < {CFG.NOISE_FLOOR_SYNTHETIC_MAX}). "
            f"Real camera sensors always introduce photon shot noise and "
            f"read noise. AI generators produce pixel-perfect smoothness."
        )

    # Check spatial noise uniformity — real sensor noise is spatially uniform
    # AI images often have non-uniform noise (smooth in some areas, textured in others)
    h, w = noise.shape
    blocks = [noise[y:y+32, x:x+32].std()
              for y in range(0, h - 32, 32)
              for x in range(0, w - 32, 32)]
    if len(blocks) > 4:
        spatial_cv = float(np.std(blocks) / (np.mean(blocks) + 1e-9))
        if spatial_cv > 0.8:
            risk += CFG.WEIGHTS["noise_floor"] // 2
            reasons.append(
                f"[NOISE] Spatially non-uniform noise (CV={spatial_cv:.2f}). "
                f"Extreme variation between smooth and textured regions suggests "
                f"AI-generated texture synthesis or inpainting."
            )

    return {
        "score": min(risk, 100),
        "confidence": 0.80,
        "is_strong": risk >= 20,
        "reasons": reasons,
        "debug": {"noise_std": noise_std, "noise_kurtosis": noise_kurt}
    }


# ─────────────────────────────────────────────────────────────────
#  Detector 6 — DCT Block Grid Detection (GAN / Upscaler Artifacts)
# ─────────────────────────────────────────────────────────────────
def detect_dct_grid(img_bgr: np.ndarray) -> dict:
    """
    GAN generators and JPEG upscalers leave periodic 8x8 or 16x16 block
    grid artifacts detectable via FFT peak/mean ratio and boundary analysis.
    """
    risk = 0
    reasons = []

    gray = _to_gray(img_bgr).astype(np.float32)
    h, w = gray.shape

    # ── FFT grid peak detection ──────────────────────────────────
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    # Zero out DC component
    cy, cx = h // 2, w // 2
    magnitude[cy - 2:cy + 3, cx - 2:cx + 3] = 0

    peak = float(magnitude.max())
    mean_val = float(magnitude.mean())
    ratio = peak / (mean_val + 1e-10)

    if ratio > CFG.DCT_GRID_PEAK_RATIO:
        risk += CFG.WEIGHTS["dct_grid"]
        reasons.append(
            f"[DCT-GRID] Structural frequency grid detected — "
            f"FFT peak/mean ratio={ratio:.1f} "
            f"(threshold > {CFG.DCT_GRID_PEAK_RATIO}). "
            f"Definitive signature of GAN generation grids or bilinear upscaler artifacts."
        )

    # ── Block boundary mismatch ──────────────────────────────────
    # Check 8x8 JPEG block boundaries for inconsistency
    if h > 64 and w > 64:
        block_diffs = []
        for y in range(8, h - 8, 8):
            row_diff = float(np.mean(np.abs(gray[y, :] - gray[y - 1, :])))
            block_diffs.append(row_diff)
        for x in range(8, w - 8, 8):
            col_diff = float(np.mean(np.abs(gray[:, x] - gray[:, x - 1])))
            block_diffs.append(col_diff)

        if block_diffs:
            all_diffs = []
            for y in range(1, h - 1):
                all_diffs.append(float(np.mean(np.abs(gray[y, :] - gray[y - 1, :]))))

            boundary_mean = float(np.mean(block_diffs))
            global_mean = float(np.mean(all_diffs))
            mismatch_ratio = boundary_mean / (global_mean + 1e-9) - 1.0

            if mismatch_ratio > CFG.DCT_MISMATCH_THRESHOLD:
                risk += CFG.WEIGHTS["dct_mismatch"]
                reasons.append(
                    f"[DCT-GRID] Block boundary discontinuity — mismatch_ratio={mismatch_ratio:.3f} "
                    f"(threshold > {CFG.DCT_MISMATCH_THRESHOLD}). "
                    f"Indicates JPEG re-encoding mismatch or AI inpainting block artifacts."
                )

    return {
        "score": min(risk, 100),
        "confidence": 0.90,
        "is_strong": ratio > CFG.DCT_GRID_PEAK_RATIO,
        "reasons": reasons,
        "debug": {"fft_peak_ratio": ratio}
    }


# ─────────────────────────────────────────────────────────────────
#  Detector 7 — Copy-Move Forgery Detection
# ─────────────────────────────────────────────────────────────────
def detect_copy_move(img_bgr: np.ndarray) -> dict:
    """
    Detect copy-move forgeries using SIFT keypoint matching.
    Regions copied and pasted within the same image share identical features.
    """
    risk = 0
    reasons = []

    gray = _to_gray(img_bgr)
    h, w = gray.shape

    if h < 100 or w < 100:
        return {"score": 0, "confidence": 0.1, "reasons": ["Image too small for copy-move detection."]}

    try:
        sift = cv2.SIFT_create(nfeatures=500)
        kp, des = sift.detectAndCompute(gray, None)

        if des is None or len(kp) < 20:
            return {"score": 0, "confidence": 0.2, "reasons": ["Insufficient keypoints for copy-move detection."]}

        # Match descriptors against themselves
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des, des, k=3)

        # Find matches that are NOT the same keypoint but have very similar descriptors
        suspicious = []
        for m_list in matches:
            if len(m_list) < 2:
                continue
            for m in m_list[1:]:  # Skip self-match (index 0)
                if m.distance < 0.6 * m_list[0].distance:
                    continue
                pt1 = kp[m.queryIdx].pt
                pt2 = kp[m.trainIdx].pt
                dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                if dist > 20:  # Must be spatially separated
                    suspicious.append((pt1, pt2, m.distance))

        if len(suspicious) >= CFG.COPY_MOVE_MIN_MATCHES:
            risk += CFG.WEIGHTS["copy_move"]
            reasons.append(
                f"[COPY-MOVE] {len(suspicious)} suspicious keypoint pairs detected "
                f"(threshold >= {CFG.COPY_MOVE_MIN_MATCHES}). "
                f"Spatially separated regions share near-identical features — "
                f"characteristic of copy-paste or clone-stamp manipulation."
            )

    except Exception as e:
        return {"score": 0, "confidence": 0.1, "reasons": [f"Copy-move detection failed: {e}"]}

    return {
        "score": min(risk, 100),
        "confidence": 0.70,
        "is_strong": risk >= 35,
        "reasons": reasons,
        "debug": {"suspicious_pairs": len(suspicious) if "suspicious" in dir() else 0}
    }


# ─────────────────────────────────────────────────────────────────
#  Detector 8 — AI Resolution Fingerprint
# ─────────────────────────────────────────────────────────────────
def detect_ai_resolution(img_bgr: np.ndarray) -> dict:
    """
    AI generators produce images at fixed output resolutions.
    Organic camera photos rarely match these exact dimensions.
    """
    h, w = img_bgr.shape[:2]
    dims = (w, h)

    if dims in CFG.AI_RESOLUTIONS:
        return {
            "score": CFG.WEIGHTS["ai_resolution"],
            "confidence": 0.60,
            "is_strong": False,
            "reasons": [
                f"[RESOLUTION] Exact AI generator output size detected — {w}×{h}px. "
                f"This is a known default output resolution for diffusion model pipelines. "
                f"Combine with other signals for higher confidence."
            ]
        }

    return {"score": 0, "confidence": 0.9, "is_strong": False, "reasons": []}


# ─────────────────────────────────────────────────────────────────
#  Detector 9 — Metadata Forensics
# ─────────────────────────────────────────────────────────────────
def detect_metadata_anomalies(image_path: str) -> dict:
    """
    Inspect EXIF, XMP, and file headers for AI generation signatures
    and suspicious metadata patterns.
    """
    risk = 0
    reasons = []

    if not HAS_PIL:
        return {"score": 0, "confidence": 0, "reasons": ["PIL not available for metadata inspection."]}

    try:
        img_pil = PILImage.open(image_path)
        info = img_pil.info or {}
        exif_data = {}

        # ── EXIF ────────────────────────────────────────────────
        if hasattr(img_pil, "_getexif") and img_pil._getexif():
            raw = img_pil._getexif()
            exif_data = {k: str(v).lower() for k, v in (raw or {}).items()}

        all_text = " ".join(str(v).lower() for v in {**info, **exif_data}.values())

        # Check for AI software signatures
        matched_tags = [tag for tag in CFG.AI_SOFTWARE_TAGS if tag in all_text]
        if matched_tags:
            risk += CFG.WEIGHTS["metadata_ai"]
            reasons.append(
                f"[METADATA] AI generation software signatures found: {matched_tags}. "
                f"Embedded metadata directly references known AI image generation pipelines."
            )

        # ── C2PA / CAI provenance check ──────────────────────────
        with open(image_path, "rb") as f:
            raw_bytes = f.read(min(1024 * 64, 65536))  # Read first 64KB
        has_c2pa = b"c2pa" in raw_bytes.lower() or b"cai\x00" in raw_bytes
        if not has_c2pa and img_pil.format == "JPEG":
            risk += CFG.WEIGHTS["c2pa_missing"]
            reasons.append(
                "[METADATA] No C2PA/CAI provenance data found. "
                "Modern cameras and trusted editing software embed content credentials. "
                "Absence is a weak but notable indicator."
            )

        # ── Missing EXIF on JPEG ─────────────────────────────────
        if not exif_data and img_pil.format == "JPEG":
            risk += CFG.WEIGHTS["metadata_no_exif"]
            reasons.append(
                "[METADATA] No EXIF data in JPEG. "
                "AI-generated JPEGs typically strip or omit camera metadata. "
                "Real camera photos almost always contain EXIF (camera model, GPS, datetime)."
            )

        # ── PNG text chunks (SD prompt injection check) ──────────
        if img_pil.format == "PNG":
            png_meta = {k.lower(): str(v).lower()
                        for k, v in img_pil.text.items()} if hasattr(img_pil, "text") else {}
            ai_keys = {"parameters", "prompt", "negative prompt", "cfg scale",
                       "sampler", "seed", "model", "steps"}
            found_keys = ai_keys & set(png_meta.keys())
            if found_keys:
                risk += CFG.WEIGHTS["metadata_ai"]
                reasons.append(
                    f"[METADATA] AI generation parameters embedded in PNG metadata: "
                    f"{found_keys}. Stable Diffusion and ComfyUI embed prompts and "
                    f"generation settings directly into PNG tEXt chunks."
                )

    except Exception as e:
        return {"score": 0, "confidence": 0.1, "reasons": [f"Metadata analysis failed: {e}"]}

    return {
        "score": min(risk, 100),
        "confidence": 0.90,
        "is_strong": risk >= 30,
        "reasons": reasons
    }


# ─────────────────────────────────────────────────────────────────
#  Master entry point
# ─────────────────────────────────────────────────────────────────
def analyze_image(image_path: str, is_video_frame: bool = False) -> dict:
    """
    Run full image forensics pipeline.
    """
    try:
        img = _load_bgr(image_path)
    except Exception as e:
        return {"score": 0, "confidence": 0, "is_strong": False,
                "reasons": [f"Image load failed: {e}"], "sub_scores": {}}

    h, w = img.shape[:2]
    if h < 32 or w < 32:
        return {"score": 10, "confidence": 0.1, "is_strong": False,
                "reasons": ["Image too small for reliable forensic analysis."],
                "sub_scores": {}}

    # ── Run all detectors ────────────────────────────────────────
    detectors = {
        "ela":            detect_ela(img, image_path),
        "srm":            detect_srm_residuals(img),
        "spectral_slope": detect_spectral_slope(img),
        "chroma":         detect_chromatic_aberration(img),
        "noise_floor":    detect_noise_floor(img),
        "dct_grid":       detect_dct_grid(img),
        "copy_move":      detect_copy_move(img),
        "ai_resolution":  detect_ai_resolution(img),
        "metadata":       detect_metadata_anomalies(image_path),
    }

    all_reasons = []
    
    # Context-Aware Threshold Filtering for Video Frames
    if is_video_frame:
        # Video compression (H.264) produces grids, artifacts, and destroys noise floors.
        # We must ignore these technical signals unless they are EXTREMELY strong.
        
        # 1. Disable Chroma perfect alignment check for video
        chroma_res = detectors["chroma"]
        if chroma_res["score"] > 0:
            chroma_res["score"] = 0
            chroma_res["is_strong"] = False
            chroma_res["reasons"] = []

        # 2. Huge threshold for Copy-Move
        cm_res = detectors["copy_move"]
        if cm_res["score"] > 0:
            pair_count = cm_res.get("debug", {}).get("suspicious_pairs", 0)
            if pair_count < 600:
                cm_res["score"] = 0
                cm_res["is_strong"] = False
                cm_res["reasons"] = []

        # 3. Huge threshold for DCT Grid (ignore most H.264 compression grids)
        dct_res = detectors["dct_grid"]
        if dct_res["score"] > 0:
            ratio = dct_res.get("debug", {}).get("fft_peak_ratio", 0)
            if ratio < 500: # Standard H.264 is often 150-300
                dct_res["score"] = 0
                dct_res["is_strong"] = False
                dct_res["reasons"] = []

        # 4. Suppress ELA/SRM/Spectral Slope for video (too noisy)
        for k in ["ela", "srm", "spectral_slope", "noise_floor"]:
            detectors[k]["score"] = 0
            detectors[k]["is_strong"] = False
            detectors[k]["reasons"] = []

    for d in detectors.values():
        all_reasons.extend(d.get("reasons", []))

    # ── Confidence-weighted max-biased fusion ────────────────────
    strong = [d for d in detectors.values()
              if d.get("is_strong") and d.get("confidence", 0) > 0.5]

    if strong:
        final_score = max(d["score"] for d in strong)
        # Corroborating signals add a bounded boost
        corroboration = sum(
            d["score"] * d.get("confidence", 0.5) * 0.10
            for d in detectors.values()
            if not d.get("is_strong") and d.get("score", 0) > 10
        )
        final_score = min(final_score + corroboration, 100)
    else:
        weighted = [
            (d["score"] * d.get("confidence", 0.5), d.get("confidence", 0.5))
            for d in detectors.values()
            if d.get("confidence", 0) > 0.3 and d.get("score", 0) > 0
        ]
        if weighted:
            scores, confs = zip(*weighted)
            final_score = sum(scores) / sum(confs)
        else:
            final_score = 10.0

    overall_confidence = float(
        np.mean([d.get("confidence", 0.5) for d in detectors.values()])
    )

    # Flatten some key indicators into metrics for easier consumption
    metrics = {
        "structural_artifact": detectors["dct_grid"].get("is_strong", False),
        "metadata_ai": detectors["metadata"].get("score", 0) >= 30,
        "copy_move": detectors["copy_move"].get("is_strong", False),
        "spectral_slope": detectors["spectral_slope"].get("debug", {}).get("spectral_slope", 0)
    }

    return {
        "score":      round(min(final_score, 100), 1),
        "confidence": round(overall_confidence, 3),
        "is_strong":  len(strong) > 0,
        "reasons":    all_reasons,
        "sub_scores": {k: {"score": v.get("score", 0),
                            "confidence": v.get("confidence", 0)}
                       for k, v in detectors.items()},
        "metrics":    metrics,
        "resolution": (w, h),
    }

def _get_face_detector():
    """Helper for video module to find faces."""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(cascade_path)
    except Exception:
        return None
