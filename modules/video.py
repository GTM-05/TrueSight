"""
TrueSight — Video forensics v3.0: p75 aggregation, FIX-5 rPPG guards, liveness contract.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Optional

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

from config import CFG

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    skimage_ssim = None  # type: ignore[misc, assignment]
from fusion.engine import compute_morphing_score
from modules.audio import analyze_audio, get_audio_envelope
from modules.image import analyze_image
from modules.metadata import check_video_metadata

log = logging.getLogger("truesight.video")


def compute_vit_video_score(per_frame_probs: list[float]) -> dict[str, Any]:
    """
    FIX-7: ViT-style frame agreement — threshold 65% (score scale), 40% of frames must fire.
    `per_frame_probs` are 0–100 style scores per frame (e.g. model output * 100).
    """
    if not per_frame_probs:
        return {
            "score": 0.0,
            "confidence": 0.1,
            "is_strong": False,
            "reasons": [],
        }

    thr = CFG.VIT_STRONG_THRESHOLD_VIDEO
    firing_ratio = sum(p > thr for p in per_frame_probs) / len(per_frame_probs)
    median_prob = float(np.median(per_frame_probs))

    if firing_ratio > CFG.VIT_MIN_FRAME_AGREEMENT:
        score = min(median_prob * 1.2, 100.0)
        is_strong = True
        reason = (
            f"[ViT] {firing_ratio:.0%} of frames exceed {thr:.0f}% threshold "
            f"(median={median_prob:.1f}%)."
        )
    else:
        score = median_prob * 0.6
        is_strong = False
        reason = (
            f"[ViT] Only {firing_ratio:.0%} of frames above {thr:.0f}% — inconsistent (median={median_prob:.1f}%)."
        )

    return {
        "score": round(score, 1),
        "confidence": min(len(per_frame_probs) / 20.0, 1.0),
        "is_strong": is_strong,
        "reasons": [reason],
    }


def _bandpass_pulse(sig: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 3) -> np.ndarray:
    if len(sig) < 16 or fs <= 0:
        return sig.astype(np.float64)
    nyq = 0.5 * fs
    lo = max(low_hz / nyq, 0.01)
    hi = min(high_hz / nyq, 0.99)
    if lo >= hi:
        return sig.astype(np.float64)
    b, a = butter(order, [lo, hi], btype="band")
    try:
        return filtfilt(b, a, sig.astype(np.float64))
    except Exception:
        return sig.astype(np.float64)


def compute_rppg(face_roi_sequence: list[np.ndarray], fps: float) -> dict[str, Any]:
    """FIX-5: CHROM rPPG with mandatory sample/face guards and SNR ≥ 3 for pulse confirmation."""
    n = len(face_roi_sequence)
    if n < CFG.MIN_LIVENESS_FACES:
        return {
            "liveness_detected": False,
            "pulse_confirmed": False,
            "pulse_anomaly": False,
            "snr": 0.0,
            "peak_freq": 0.0,
            "confidence": 0.0,
            "is_strong": False,
            "skip_reason": "Insufficient faces",
        }
    if n < CFG.MIN_LIVENESS_SIGNALS:
        return {
            "liveness_detected": False,
            "pulse_confirmed": False,
            "pulse_anomaly": False,
            "snr": 0.0,
            "peak_freq": 0.0,
            "confidence": 0.0,
            "is_strong": False,
            "skip_reason": "Insufficient samples for FFT",
        }

    r_vals: list[float] = []
    g_vals: list[float] = []
    b_vals: list[float] = []
    for f in face_roi_sequence:
        if f is None or f.size == 0:
            continue
        r_vals.append(float(np.mean(f[:, :, 2])))
        g_vals.append(float(np.mean(f[:, :, 1])))
        b_vals.append(float(np.mean(f[:, :, 0])))

    if len(r_vals) < CFG.MIN_LIVENESS_SIGNALS:
        return {
            "liveness_detected": False,
            "pulse_confirmed": False,
            "pulse_anomaly": False,
            "snr": 0.0,
            "peak_freq": 0.0,
            "confidence": 0.0,
            "is_strong": False,
            "skip_reason": "Insufficient samples for FFT",
        }

    r_arr = np.array(r_vals, dtype=np.float64)
    g_arr = np.array(g_vals, dtype=np.float64)
    b_arr = np.array(b_vals, dtype=np.float64)

    xs = 3 * r_arr - 2 * g_arr
    ys = 1.5 * r_arr + g_arr - 1.5 * b_arr
    xs_n = xs / (np.std(xs) + 1e-9)
    ys_n = ys / (np.std(ys) + 1e-9)
    pulse = xs_n - ys_n

    pulse_filt = _bandpass_pulse(
        pulse, CFG.RPPG_BANDPASS_LOW, CFG.RPPG_BANDPASS_HIGH, fps
    )

    freqs = np.fft.rfftfreq(len(pulse_filt), 1.0 / fps)
    power = np.abs(np.fft.rfft(pulse_filt)) ** 2
    peak_idx = int(np.argmax(power))
    peak_freq = float(freqs[peak_idx])
    peak_power = float(power[peak_idx])
    bg_power = float(np.mean(np.delete(power, peak_idx))) if power.size > 1 else 1e-10
    snr = peak_power / (bg_power + 1e-10)

    pulse_confirmed = (0.7 <= peak_freq <= 3.5) and (snr >= CFG.RPPG_SNR_MIN)
    pulse_anomaly = (snr < CFG.RPPG_SNR_ANOMALY) and (n >= CFG.MIN_LIVENESS_SIGNALS)
    conf = min(n / (CFG.MIN_LIVENESS_SIGNALS * 2), 1.0)
    # Strong anchor: anomaly (SNR below ANOMALY), samples already gated, conf > 0.70
    rppg_strong = pulse_anomaly and conf > 0.70

    return {
        "liveness_detected": pulse_confirmed,
        "pulse_confirmed": pulse_confirmed,
        "pulse_anomaly": pulse_anomaly,
        "snr": float(snr),
        "peak_freq": peak_freq,
        "confidence": float(conf),
        "is_strong": bool(rppg_strong),
    }


def aggregate_frame_scores(frame_scores: list[dict]) -> dict:
    """p75 + consistency; dedupe reasons by first 60 chars (spec)."""
    if not frame_scores:
        return {
            "score": 0.0,
            "confidence": 0.1,
            "is_strong": False,
            "reasons": [],
            "sub_scores": {},
        }

    scores = [float(f["score"]) for f in frame_scores]
    p75 = float(np.percentile(scores, 75))
    p50 = float(np.median(scores))
    score_std = float(np.std(scores))
    consistency = max(0.0, 1.0 - (score_std / (p75 + 1e-9)))
    final = p75 * consistency + p50 * (1.0 - consistency)

    seen: set[str] = set()
    unique: list[str] = []
    for f in frame_scores:
        for r in f.get("reasons", []) or []:
            key = r[:60]
            if key not in seen:
                seen.add(key)
                unique.append(r)

    mean_conf = float(np.mean([float(f.get("confidence", 0.5) or 0.5) for f in frame_scores]))
    
    # Aggregate sub-scores (e.g. from image analysis per frame)
    subs = {}
    for f in frame_scores:
        fsubs = f.get("sub_scores", {})
        for k, v in fsubs.items():
            if k not in subs:
                subs[k] = []
            subs[k].append(v.get("score", 0))
    
    aggregated_subs = {}
    for k, vlist in subs.items():
        aggregated_subs[k] = {"score": float(np.percentile(vlist, 75)) if vlist else 0.0}

    return {
        "score": round(final, 1),
        "confidence": round(mean_conf * consistency, 2),
        "is_strong": p75 >= 45.0 and consistency > 0.5,
        "reasons": unique,
        "sub_scores": aggregated_subs,
    }


def _get_face_detector() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _get_face_crop(frame_bgr: np.ndarray) -> dict[str, Any]:
    face_cascade = _get_face_detector()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return {"is_face": False}

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face_img = frame_bgr[y : y + h, x : x + w]
    mouth_img = frame_bgr[
        y + int(h * 0.65) : y + int(h * 0.95), x + int(w * 0.2) : x + int(w * 0.8)
    ]
    eye_img = frame_bgr[
        y + int(h * 0.15) : y + int(h * 0.45), x + int(w * 0.1) : x + int(w * 0.9)
    ]

    return {
        "is_face": True,
        "face": face_img,
        "mouth": mouth_img,
        "eyes": eye_img,
        "box": (x, y, w, h),
    }


def _analyze_mouth_aspect_ratio(mouth_bgr: Optional[np.ndarray]) -> float:
    if mouth_bgr is None or mouth_bgr.size == 0:
        return 0.0
    h, w = mouth_bgr.shape[:2]
    return h / (w + 1e-9)


def _crop_face_pair(
    f0: np.ndarray, f1: np.ndarray, cascade: cv2.CascadeClassifier
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return aligned grayscale face crops (same box on both frames) or (None, None)."""
    gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray0, 1.1, 4)
    if len(faces) == 0:
        return None, None
    x, y, w, h = sorted(faces, key=lambda t: t[2] * t[3], reverse=True)[0]
    h0, w0 = f0.shape[:2]
    h1, w1 = f1.shape[:2]
    x2, y2 = min(x + w, w0), min(y + h, h0)
    a = f0[y:y2, x:x2]
    b = f1[y : min(y + h, h1), x : min(x + w, w1)]
    if a.size == 0 or b.size == 0:
        return None, None
    side = int(CFG.SSIM_FACE_RESIZE)
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (side, side), interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (side, side), interpolation=cv2.INTER_AREA)
    return ga, gb


def detect_ssim_morphing(
    file_path: str,
    sample_indices: np.ndarray,
    count: int,
    low_resource: bool,
) -> dict[str, Any]:
    """
    Adjacent-frame SSIM inside the face box only.
    Too stable → slow GAN-like morph; too variable → splice / inconsistent warping.
    """
    empty = {
        "ssim_std": 0.0,
        "ssim_mean": 0.0,
        "anomaly": False,
        "score": 0.0,
        "pairs": 0,
        "reasons": [],
        "mode": "none",
    }
    if low_resource or skimage_ssim is None:
        empty["reasons"] = (
            ["[FACE-SSIM] Skipped (low-resource mode)."]
            if low_resource
            else ["[FACE-SSIM] scikit-image not installed."]
        )
        return empty

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return empty

    cascade = _get_face_detector()
    vals: list[float] = []
    for idx in sample_indices:
        i = int(idx)
        if i >= count - 1:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret0, f0 = cap.read()
        ret1, f1 = cap.read()
        if not ret0 or not ret1 or f0 is None or f1 is None:
            continue
        try:
            ga, gb = _crop_face_pair(f0, f1, cascade)
            if ga is None or gb is None:
                continue
            vals.append(float(skimage_ssim(ga, gb, data_range=255)))
        except Exception:
            continue
    cap.release()

    if len(vals) < int(CFG.SSIM_FACE_MIN_PAIRS):
        return {
            **empty,
            "pairs": len(vals),
            "reasons": ["[FACE-SSIM] Too few face pairs for temporal SSIM."],
        }

    std_v = float(np.std(vals))
    mean_v = float(np.mean(vals))
    stable = mean_v >= float(CFG.SSIM_FACE_STABLE_MEAN_MIN) and std_v <= float(
        CFG.SSIM_FACE_STABLE_STD_MAX
    )
    variable = std_v >= float(CFG.SSIM_FACE_VARIABLE_STD_MIN)
    score = max(
        float(CFG.SSIM_FACE_MORPH_STABLE_SCORE) if stable else 0.0,
        float(CFG.SSIM_FACE_MORPH_VARIABLE_SCORE) if variable else 0.0,
    )
    reasons: list[str] = []
    mode = "none"
    if stable:
        mode = "stable"
        reasons.append(
            f"[FACE-SSIM] Abnormally stable face texture across frames (mean={mean_v:.4f}, "
            f"std={std_v:.4f}) — possible slow neural morphing."
        )
    if variable:
        mode = "variable" if not stable else mode + "+variable"
        reasons.append(
            f"[FACE-SSIM] High face SSIM variance (std={std_v:.4f}) — splice / warping inconsistency."
        )

    return {
        "ssim_std": std_v,
        "ssim_mean": mean_v,
        "anomaly": bool(stable or variable),
        "score": float(min(100.0, score)),
        "pairs": len(vals),
        "reasons": reasons,
        "mode": mode,
    }


def detect_face_warp(
    file_path: str,
    sample_indices: np.ndarray,
    count: int,
    low_resource: bool,
) -> dict[str, Any]:
    """Optical-flow magnitude on face-aligned crops — geometric warping between frames."""
    empty: dict[str, Any] = {
        "score": 0.0,
        "mean_mag": 0.0,
        "std_mag": 0.0,
        "samples": 0,
        "reasons": [],
    }
    if low_resource:
        empty["reasons"] = ["[FACE-WARP] Skipped (low-resource mode)."]
        return empty

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return empty

    cascade = _get_face_detector()
    mags: list[float] = []
    for idx in sample_indices:
        i = int(idx)
        if i >= count - 1:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret0, f0 = cap.read()
        ret1, f1 = cap.read()
        if not ret0 or not ret1:
            continue
        ga, gb = _crop_face_pair(f0, f1, cascade)
        if ga is None or gb is None:
            continue
        try:
            pr = cv2.pyrDown(ga)
            nx = cv2.pyrDown(gb)
            win = int(CFG.FACE_WARP_FLOW_WIN_SIZE)
            if win % 2 == 0:
                win += 1
            flow = cv2.calcOpticalFlowFarneback(
                pr, nx, None, 0.5, 3, win, 3, 5, 1.2, 0
            )
            mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            mags.append(mag)
        except Exception:
            continue
    cap.release()

    if len(mags) < int(CFG.FACE_WARP_MIN_SAMPLES):
        return {**empty, "samples": len(mags)}

    mean_mag = float(np.mean(mags))
    std_mag = float(np.std(mags))
    score = 0.0
    reasons: list[str] = []
    if mean_mag > float(CFG.OPTICAL_FLOW_WARP_THRESHOLD):
        score += float(CFG.FACE_WARP_MEAN_MAG_SCORE)
        reasons.append(
            f"[FACE-WARP] High mean optical flow in face ROI ({mean_mag:.3f} > {CFG.OPTICAL_FLOW_WARP_THRESHOLD})."
        )
    if std_mag > float(CFG.FACE_WARP_FLOW_STD_THRESHOLD):
        score += float(CFG.FACE_WARP_STD_MAG_SCORE)
        reasons.append(
            f"[FACE-WARP] Erratic flow in face ROI (std={std_mag:.3f} > {CFG.FACE_WARP_FLOW_STD_THRESHOLD})."
        )

    return {
        "score": float(min(100.0, score)),
        "mean_mag": mean_mag,
        "std_mag": std_mag,
        "samples": len(mags),
        "reasons": reasons,
    }


def compute_ela_persistence(frame_ela_results: list[dict]) -> dict:
    """
    FIX-3: ELA firing consistently across frames = strong signal.
    """
    if not frame_ela_results:
        return {"score": 0.0, "confidence": 0.1, "is_strong": False, "reasons": []}

    fired_frames = [r for r in frame_ela_results if float(r.get("score", 0) or 0) > 0]
    fire_ratio = len(fired_frames) / len(frame_ela_results)

    if fire_ratio < float(CFG.ELA_PERSIST_FRAME_RATIO) * 0.6:
        return {
            "score": 10.0,
            "confidence": 0.4,
            "is_strong": False,
            "reasons": [f"[ELA] Sporadic hits ({fire_ratio:.0%} of frames)."],
        }

    scores = [float(r["score"]) for r in fired_frames]
    mean_score = float(sum(scores) / len(scores))

    if fire_ratio >= float(CFG.ELA_PERSIST_FRAME_RATIO) and mean_score >= float(
        CFG.ELA_PERSIST_MEAN_MIN
    ):
        return {
            "score": min(mean_score * 1.3, 70.0),
            "confidence": min(fire_ratio, 0.85),
            "is_strong": fire_ratio >= float(CFG.ELA_PERSIST_STRONG_RATIO)
            and mean_score >= 25,
            "reasons": [
                f"[ELA-PERSIST] ELA anomaly in {fire_ratio:.0%} of frames "
                f"(mean={mean_score:.1f}). Systematic face-region manipulation."
            ],
        }

    return {
        "score": mean_score * fire_ratio * 1.5,
        "confidence": fire_ratio * 0.7,
        "is_strong": False,
        "reasons": [f"[ELA] Moderate persistence ({fire_ratio:.0%} frames)."],
    }


def detect_face_boundary_blend(frame: np.ndarray, face_box: Optional[tuple]) -> dict:
    """
    FIX-4: Face-swap blending seam detection via Laplacian edge analysis.
    """
    if face_box is None:
        return {"score": 0.0, "confidence": 0.0, "is_strong": False, "reasons": []}

    x, y, w, h = face_box
    img_h, img_w = frame.shape[:2]

    BORDER = 8
    x1, y1 = max(x - BORDER, 0), max(y - BORDER, 0)
    x2, y2 = min(x + w + BORDER, img_w), min(y + h + BORDER, img_h)

    outer_region = frame[y1:y2, x1:x2]
    inner_region = frame[y : y + h, x : x + w]

    if outer_region.size == 0 or inner_region.size == 0:
        return {"score": 0.0, "confidence": 0.1, "is_strong": False, "reasons": []}

    gray_outer = cv2.cvtColor(outer_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_inner = cv2.cvtColor(inner_region, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lap_inner = float(cv2.Laplacian(gray_inner, cv2.CV_32F).var())
    lap_outer = float(cv2.Laplacian(gray_outer, cv2.CV_32F).var())

    if lap_outer < 1.0:
        return {"score": 0.0, "confidence": 0.2, "is_strong": False, "reasons": []}

    blend_ratio = lap_inner / (lap_outer + 1e-9)
    risk = 0.0
    reasons = []

    if blend_ratio > float(CFG.BLEND_RATIO_HIGH):
        risk = 35.0
        reasons.append(
            f"[BLEND] Face boundary blending seam (ratio={blend_ratio:.2f} > {CFG.BLEND_RATIO_HIGH})."
        )
    elif blend_ratio < float(CFG.BLEND_RATIO_LOW) and lap_inner < 50:
        risk = 30.0
        reasons.append(
            f"[BLEND] Over-smoothed face interior (ratio={blend_ratio:.2f} < {CFG.BLEND_RATIO_LOW})."
        )
    elif blend_ratio > float(CFG.BLEND_MILD_RATIO):
        risk = 18.0
        reasons.append(f"[BLEND] Mild boundary discontinuity (ratio={blend_ratio:.2f}).")

    return {
        "score": risk,
        "confidence": 0.70,
        "is_strong": risk >= 35.0,
        "reasons": reasons,
    }


def detect_color_inconsistency(frame: np.ndarray, face_box: Optional[tuple]) -> dict:
    """
    NEW: Face-swap colour temperature mismatch (Delta-E).
    """
    if face_box is None:
        return {"score": 0.0, "confidence": 0.0, "is_strong": False, "reasons": []}

    x, y, w, h = face_box
    img_h, img_w = frame.shape[:2]

    fx1, fy1 = x + int(w * 0.2), y + int(h * 0.2)
    fx2, fy2 = x + int(w * 0.8), y + int(h * 0.8)
    face_roi = frame[fy1:fy2, fx1:fx2]

    neck_y1 = min(y + h, img_h - 20)
    neck_y2 = min(y + h + int(h * 0.4), img_h)
    neck_x1, neck_x2 = x + int(w * 0.2), x + int(w * 0.8)
    neck_roi = frame[neck_y1:neck_y2, neck_x1:neck_x2]

    if face_roi.size == 0 or neck_roi.size == 0:
        return {"score": 0.0, "confidence": 0.1, "is_strong": False, "reasons": []}

    face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    neck_lab = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2Lab).astype(np.float32)

    # NEW: Skin-guard. Skip if neck ROI is too gray/dark (common for clothing or shadows)
    face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    neck_hsv = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2HSV)
    face_sat = np.mean(face_hsv[:,:,1])
    neck_sat = np.mean(neck_hsv[:,:,1])
    
    if face_sat < CFG.COLOR_SKIN_THRESHOLD_MIN or neck_sat < CFG.COLOR_SKIN_THRESHOLD_MIN:
        return {"score": 0.0, "confidence": 0.2, "is_strong": False, "reasons": []}

    # REFINED: Chromaticity-only Delta-E (ignore 'L' channel for shadow tolerance)
    face_chroma = face_lab.reshape(-1, 3)[:, 1:] # a, b only
    neck_chroma = neck_lab.reshape(-1, 3)[:, 1:] # a, b only
    face_mean = np.mean(face_chroma, axis=0)
    neck_mean = np.mean(neck_chroma, axis=0)
    delta_e = float(np.sqrt(np.sum((face_mean - neck_mean) ** 2)))

    risk = 0.0
    reasons = []

    if delta_e > float(CFG.COLOR_DELTA_E_STRONG):
        risk = 40.0
        reasons.append(
            f"[COLOR] Strong face/neck colour mismatch (Delta-E={delta_e:.1f} > {CFG.COLOR_DELTA_E_STRONG})."
        )
    elif delta_e > float(CFG.COLOR_DELTA_E_MEDIUM):
        risk = 25.0
        reasons.append(
            f"[COLOR] Moderate face/neck mismatch (Delta-E={delta_e:.1f} > {CFG.COLOR_DELTA_E_MEDIUM})."
        )

    return {
        "score": risk,
        "confidence": 0.75,
        "is_strong": risk >= 40.0,
        "reasons": reasons,
    }


def analyze_video(
    file_path: str, low_resource: bool = False, deep_scan: bool = False
) -> dict:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return {
            "score": 0.0,
            "confidence": 0.0,
            "is_strong": False,
            "reasons": ["Unreadable video."],
            "sub_scores": {},
            "metrics": {},
            "liveness": {},
            "ai_gen_score": 0,
            "manip_score": 0,
            "morphing_score": 0.0,
            "morph_components": {"ssim_morph": 0.0, "face_warp": 0.0},
            "ssim": {},
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        cap.release()
        return {
            "score": 0.0,
            "confidence": 0.0,
            "is_strong": False,
            "reasons": ["Empty video."],
            "sub_scores": {},
            "metrics": {},
            "liveness": {},
            "ai_gen_score": 0,
            "manip_score": 0,
            "morphing_score": 0.0,
            "morph_components": {"ssim_morph": 0.0, "face_warp": 0.0},
            "ssim": {},
        }

    if deep_scan:
        num_samples = 45
    elif low_resource:
        num_samples = 5
    else:
        num_samples = 30
    sample_indices = np.linspace(0, max(count - 1, 0), num_samples, dtype=int)

    frame_results: list[dict] = []
    ela_frame_results: list[dict] = []
    blend_results: list[dict] = []
    color_results: list[dict] = []
    face_roi_sequence: list[np.ndarray] = []
    mar_values: list[float] = []
    eye_vars: list[float] = []

    tmp_dir = tempfile.mkdtemp()
    temp_frames: list[str] = []

    for i, idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        f_path = os.path.join(tmp_dir, f"frame_{i}.jpg")
        cv2.imwrite(f_path, frame)
        temp_frames.append(f_path)

        img_res = analyze_image(f_path, source="video")
        frame_results.append(img_res)
        if "ela" in img_res:
            ela_frame_results.append(img_res["ela"])
        elif img_res.get("metrics", {}).get("ela_std_dev"):
            # Fallback if ela sub-dict not present
            ela_frame_results.append(
                {"score": img_res["score"] if "ELA" in str(img_res.get("reasons")) else 0}
            )

        roi = _get_face_crop(frame)
        if roi["is_face"] and roi["face"].size > 0:
            face_roi_sequence.append(roi["face"].copy())
            mar_values.append(_analyze_mouth_aspect_ratio(roi["mouth"]))
            e = roi["eyes"]
            if e is not None and e.size > 0:
                eye_vars.append(float(np.std(cv2.cvtColor(e, cv2.COLOR_BGR2GRAY))))
            else:
                eye_vars.append(0.0)

            # FIX-4: Face blend + NEW: Color inconsistency
            box = roi.get("box")
            blend_r = detect_face_boundary_blend(frame, box)
            if blend_r["score"] > 0:
                blend_results.append(blend_r)

            color_r = detect_color_inconsistency(frame, box)
            if color_r["score"] > 0:
                color_results.append(color_r)

    cap.release()

    # FIX-3: ELA Persistence
    ela_persist = compute_ela_persistence(ela_frame_results)

    # Aggregate Blend & Color
    def _agg_signals(results: list[dict], tag: str) -> dict:
        if not results:
            return {"score": 0.0, "confidence": 0.3, "is_strong": False, "reasons": []}
        scores = [float(r["score"]) for r in results]
        return {
            "score": float(np.percentile(scores, 75)),
            "confidence": min(len(results) / 10.0, 0.85),
            "is_strong": any(r.get("is_strong") for r in results),
            "reasons": results[0]["reasons"] if results[0].get("reasons") else [],
        }

    blend_agg = _agg_signals(blend_results, "BLEND")
    color_agg = _agg_signals(color_results, "COLOR")

    morph_ssim = detect_ssim_morphing(file_path, sample_indices, count, low_resource)
    morph_warp = detect_face_warp(file_path, sample_indices, count, low_resource)

    agg = aggregate_frame_scores(frame_results)

    rppg = compute_rppg(face_roi_sequence, fps)
    liveness_reasons: list[str] = []
    if rppg.get("skip_reason"):
        liveness_reasons.append(
            f"[LIVENESS] rPPG skipped: {rppg['skip_reason']} (faces={len(face_roi_sequence)})."
        )
    elif rppg.get("pulse_anomaly") and rppg.get("is_strong"):
        liveness_reasons.append(
            f"[LIVENESS] Pulse signal anomaly (SNR={rppg.get('snr', 0):.2f}). Potential deepfake."
        )

    iris_jitter = float(np.std(eye_vars)) if len(eye_vars) > 2 else 0.0
    blink_proxy = sum(1 for m in mar_values if m < 0.25) if mar_values else 0

    liveness_block: dict[str, Any] = {
        "liveness_detected": bool(rppg.get("liveness_detected")),
        "pulse_confirmed": bool(rppg.get("pulse_confirmed")),
        "pulse_anomaly": bool(rppg.get("pulse_anomaly")),
        "blink_count": int(min(blink_proxy, 10)),
        "iris_jitter": iris_jitter,
        "confidence": float(rppg.get("confidence", 0.0) or 0.0),
        "snr": float(rppg.get("snr", 0.0) or 0.0),
        "peak_freq": float(rppg.get("peak_freq", 0.0) or 0.0),
        "skip_reason": rppg.get("skip_reason"),
    }

    audio_data: dict = {"score": 0.0, "reasons": []}
    audio_path = os.path.join(tmp_dir, "audio.wav")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                file_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                audio_path,
                "-y",
            ],
            capture_output=True,
            check=True,
        )
        audio_data = analyze_audio(audio_path)
        if not low_resource and len(mar_values) > 10:
            env = get_audio_envelope(audio_path, num_points=len(mar_values))
            corr = float(np.corrcoef(mar_values, env)[0, 1])
            if corr < 0.1:
                agg["score"] = float(agg["score"]) + 15.0
                agg["reasons"].append(f"[AV-SYNC] Weak lip-audio correlation ({corr:.2f}).")
    except Exception:
        pass

    meta_res = check_video_metadata(file_path)

    # Combine all reasons
    final_reasons = set(agg.get("reasons") or [])
    final_reasons.update(liveness_reasons)
    final_reasons.update(audio_data.get("reasons") or [])
    final_reasons.update(meta_res.get("reasons") or [])
    final_reasons.update(ela_persist.get("reasons") or [])
    final_reasons.update(blend_agg.get("reasons") or [])
    final_reasons.update(color_agg.get("reasons") or [])
    final_reasons.update(morph_ssim.get("reasons") or [])
    final_reasons.update(morph_warp.get("reasons") or [])

    final_reasons_list = sorted(list(filter(None, final_reasons)))

    # Calculate final scores
    frame_scores = [float(f.get("score", 0) or 0) for f in frame_results]
    ai_gen_score = int(round(float(np.percentile(frame_scores, 75)))) if frame_scores else 0

    video_for_morph = {
        "metrics": {"meta_score": float(meta_res.get("score", 0) or 0)},
        "morph_components": {
            "ssim_morph": float(morph_ssim.get("score", 0) or 0),
            "face_warp": float(morph_warp.get("score", 0) or 0),
            "color_score": float(color_agg.get("score", 0) or 0),
        },
    }
    morphing_score = float(compute_morphing_score(video_for_morph, audio_data))
    manip_score = int(round(morphing_score))

    # Clean up
    for f in temp_frames:
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError:
            pass
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # Final score integration with Cross-Detector Consensus (CDC) Boost
    srm_s = agg.get("sub_scores", {}).get("srm", {}).get("score", 0)
    noise_s = agg.get("sub_scores", {}).get("noise", {}).get("score", 0)
    
    # Liveness skip as a semantic indicator (common in deepfakes)
    live_skip_s = 15.0 if liveness_block.get("skip_reason") else 0.0

    detector_scores = {
        "AI": ai_gen_score,
        "Morphing": manip_score,
        "ELA": ela_persist.get("score", 0),
        "Blend": blend_agg.get("score", 0),
        "Color": color_agg.get("score", 0),
        "SRM": srm_s,
        "Metadata": meta_res.get("score", 0),
        "Audio": audio_data.get("score", 0),
        "Noise": noise_s,
        "Liveness-Skip": live_skip_s,
    }
    
    # Identify firing detectors
    # Binary/Metadata signals trigger at 9.0+, continuous signals at 15.0+
    firing = []
    for name, val in detector_scores.items():
        if name in ["Metadata", "Noise", "Liveness-Skip", "Audio"]:
            if val >= 9.0: firing.append(name)
        else:
            if val >= 15.0: firing.append(name)
    
    base_final = max(detector_scores.values())
    final_score = base_final
    
    # CONSENSUS BOOST: 4+ independent sectors = High Risk
    # IF 3 sectors fire, we only boost if at least one is "strong" (anchored)
    is_any_strong = bool(
        agg.get("is_strong") or ela_persist.get("is_strong") or 
        blend_agg.get("is_strong") or color_agg.get("is_strong") or 
        rppg.get("is_strong") or 
        morphing_score >= CFG.MORPHING_IS_STRONG_THRESHOLD
    )
    
    # NEW: Only allow large boost if at least one 'structural' detector fired
    structural_list = ["AI", "ELA", "SRM", "Morphing", "Blend"]
    is_structural = any(s in firing for s in structural_list)

    use_boost = False
    if len(firing) >= int(CFG.CONSENSUS_MIN_DETECTORS) and is_structural:
        use_boost = True
    elif len(firing) >= 3 and CFG.CONSENSUS_REQUIRED_STRONG and is_any_strong and is_structural:
        use_boost = True

    if use_boost:
        # Accumulative agreement boost (TARGET: >80% for deepfakes)
        boosted = base_final + float(CFG.CONSENSUS_BOOST_ADDITIVE)
        final_score = max(boosted, float(CFG.CONSENSUS_SCORE_FLOOR))
        final_reasons_list.append(
            f"[CONSENSUS] {len(firing)} independent forensic sectors firing ({', '.join(firing)}). "
            f"Cross-verification confirms high-risk manipulation."
        )

    final_score = float(min(100.0, final_score))

    return {
        "score": final_score,
        "final_score": final_score, # For LLM compatibility
        "ai_gen_score": ai_gen_score,
        "manip_score": manip_score,
        "morphing_score": round(morphing_score, 1),
        "confidence": float(agg.get("confidence", 0.5)),
        "is_strong": bool(
            agg.get("is_strong", False)
            or ela_persist.get("is_strong")
            or blend_agg.get("is_strong")
            or color_agg.get("is_strong")
            or rppg.get("is_strong")
            or len(firing) >= 4
        ),
        "reasons": final_reasons_list,
        "sub_scores": {k: {"score": v} for k, v in detector_scores.items()},
        "liveness": liveness_block,
        "metadata": meta_res,
        "audio": audio_data,
        "sub_detectors": {
            "ela_persistence": ela_persist,
            "face_blend": blend_agg,
            "colour_mismatch": color_agg,
            "ssim_morphing": morph_ssim,
            "face_warp": morph_warp,
        },
        "metrics": {
            "num_frames": len(frame_results),
            "rppg_snr": rppg.get("snr", 0),
            "iris_jitter": iris_jitter,
        },
    }
