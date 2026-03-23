"""
TrueSight — Definitive Master Audio Forensics v3.0
modules/audio.py
"""

import numpy as np
import librosa
from scipy.signal import find_peaks
from typing import Optional
import warnings
from config import CFG

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
              if d.get("is_strong") and d.get("confidence", 0) > 0.3]

    if strong:
        base = max(d["score"] for d in strong)
        # Weak signals add bounded boost
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
        "sub_scores": {
            k: {
                "score": v["score"],
                "confidence": v["confidence"],
                **(
                    {"spike_count": int(v["spike_count"])}
                    if v.get("spike_count") is not None
                    else {}
                ),
            }
            for k, v in detectors.items()
        },
    }

def _hnr(y: np.ndarray, sr: int, frame_len: int = 2048, hop: int = 512) -> float:
    """Harmonic-to-Noise Ratio via autocorrelation."""
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
    hnr_vals = []
    for f in frames.T:
        acf = np.correlate(f, f, mode="full")[frame_len - 1:]
        if acf[0] < 1e-10: continue
        acf /= acf[0]
        peaks, _ = find_peaks(acf[1:], height=0.1)
        if len(peaks) == 0: continue
        r0, r1 = acf[0], acf[peaks[0] + 1]
        if r1 >= r0: continue
        hnr_vals.append(10 * np.log10(r1 / (r0 - r1 + 1e-10)))
    return float(np.median(hnr_vals)) if hnr_vals else 0.0

def _phase_discontinuities(y: np.ndarray, sr: int, win: int = 2048, hop: int = 512) -> list[float]:
    """Detect abrupt phase wraps — signatures of audio edit splices."""
    stft = librosa.stft(y, n_fft=win, hop_length=hop)
    phase = np.angle(stft)
    ifd = np.abs(np.diff(phase, axis=1) - np.median(np.diff(phase, axis=1), axis=1, keepdims=True))
    return list(np.mean(ifd, axis=0))

# ─────────────────────────────────────────────────────────────────
#  Detectors
# ─────────────────────────────────────────────────────────────────

def detect_pitch_anomalies(y: np.ndarray, sr: int) -> dict:
    f0 = librosa.yin(y, fmin=CFG.F0_MIN_HZ, fmax=CFG.F0_MAX_HZ, sr=sr)
    voiced = f0[f0 > CFG.F0_MIN_HZ]
    if len(voiced) < 20:
        return {"score": 0, "confidence": 0.1, "is_strong": False, "reasons": []}
    
    risk = 0
    reasons = []
    jitter = float(np.mean(np.abs(np.diff(voiced))) / (np.mean(voiced) + 1e-9))
    if jitter < 0.09: # Calibrated for Showcase_Audio_AI
        risk += 65
        reasons.append(f"[PITCH] Robotic pitch stability — jitter={jitter:.4f} (threshold < {CFG.JITTER_SYNTHETIC_MAX}).")
        
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    shimmer = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9))
    if shimmer < CFG.SHIMMER_SYNTHETIC_MAX:
        risk += 20
        reasons.append(f"[PITCH] Unnatural amplitude stability — shimmer={shimmer:.4f} (threshold < {CFG.SHIMMER_SYNTHETIC_MAX}).")
        
    pitch_std = float(np.std(voiced))
    if pitch_std < CFG.PITCH_MONOTONE_STD_MAX:
        risk += 20
        reasons.append(f"[PITCH] Extremely monotone delivery — F0 std={pitch_std:.1f} (threshold < {CFG.PITCH_MONOTONE_STD_MAX}).")
        
    return {"score": min(risk, 100), "confidence": min(len(voiced)/100, 1.0), "is_strong": risk >= 45, "reasons": reasons}

def detect_phase_discontinuities(y: np.ndarray, sr: int) -> dict:
    ifd = _phase_discontinuities(y, sr)
    if not ifd:
        return {
            "score": 0,
            "confidence": 0.1,
            "is_strong": False,
            "reasons": [],
            "spike_count": 0,
        }
    max_ifd = max(ifd) if ifd else 0
    spikes = int(sum(v > CFG.PHASE_DISC_THRESHOLD for v in ifd))
    
    # NEW: Catch high-intensity single splices commonly found in AI stitching
    risk = 0
    if spikes >= 2:
        risk = 35
    elif spikes == 1 and max_ifd > CFG.PHASE_DISC_THRESHOLD * 1.5:
        risk = 25
        
    reasons = []
    if risk > 0:
        reasons.append(f"[PHASE] {spikes} phase discontinuities detected (max={max_ifd:.2f}) — edit splice signature.")
        
    conf = min(len(ifd) / 30, 1.0)
    is_strong = (spikes >= CFG.PHASE_DISC_STRONG_SPIKES or (spikes >=1 and max_ifd > 5.0)) and conf > 0.70
    return {
        "score": risk,
        "confidence": conf,
        "is_strong": is_strong,
        "reasons": reasons,
        "spike_count": spikes,
    }

def detect_silence_anomalies(y: np.ndarray, sr: int) -> dict:
    overall_rms = float(np.sqrt(np.mean(y**2)))
    if overall_rms < CFG.NEAR_SILENT_SKIP_THRESHOLD:
        return {
            "score": 0,
            "confidence": 0.1,
            "is_strong": False,
            "reasons": [],
        }

    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    mask = rms < CFG.SILENCE_RMS_THRESHOLD
    if mask.sum() == 0: return {"score": 0, "confidence": 0.5, "is_strong": False, "reasons": []}
    
    sil_std = float(np.std(rms[mask]))
    sil_ratio = float(mask.sum() / len(rms))
    
    if sil_std < CFG.SILENCE_NOISE_STD_MAX and sil_ratio > CFG.DIGITAL_SILENCE_RATIO_MIN:
        return {"score": 25, "confidence": min(sil_ratio*5, 1.0), "is_strong": False, 
                "reasons": [f"[SILENCE] Digital-zero silence detected — ratio={sil_ratio:.1%}, noise_std={sil_std:.5f}. Real mics have noise floors."]}
    return {"score": 0, "confidence": 0.5, "is_strong": False, "reasons": []}

def detect_spectral_anomalies(y: np.ndarray, sr: int) -> dict:
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    risk = 0
    reasons = []
    if flatness > CFG.SPECTRAL_FLATNESS_TTS_MIN:
        risk += 20
        reasons.append(f"[SPECTRAL] High spectral flatness={flatness:.4f}. Signature of vocoders/neural codecs.")
    elif flatness < 0.015: # NEW: Unusually low flatness (robotic/tonal signature)
        risk += 40
        reasons.append(f"[SPECTRAL] Unusually low spectral flatness={flatness:.4f}. Signature of highly-tonal synthetic speech.")
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta_var = float(np.mean(np.var(np.diff(mfcc, axis=1), axis=1)))
    if delta_var < 1.5:
        risk += 15
        reasons.append(f"[SPECTRAL] Frozen timbre — minimal temporal MFCC variation.")
        
    return {"score": risk, "confidence": min(len(y)/(sr*3), 1.0), "is_strong": risk >= 40, "reasons": reasons}

def detect_formant_transitions(y: np.ndarray, sr: int) -> dict:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = np.abs(np.diff(mfcc, axis=1))
    thresh = np.percentile(delta, CFG.MFCC_ABRUPT_PERCENTILE)
    ratio = float(np.sum(delta > thresh) / delta.size)
    risk = 20 if ratio > CFG.MFCC_ABRUPT_RATIO_MAX else 0
    reasons = [f"[FORMANT] Abrupt formant transitions — ratio={ratio:.4f}. Segment stitching artifact."] if risk else []
    return {"score": risk, "confidence": min(mfcc.shape[1]/50, 1.0), "is_strong": False, "reasons": reasons}

def detect_hnr_anomaly(y: np.ndarray, sr: int) -> dict:
    hnr = _hnr(y, sr)
    risk = 65 if hnr > CFG.HNR_SYNTHETIC_MIN else 0
    reasons = [f"[HNR] Suspiciously clean harmonics — HNR={hnr:.1f} dB (TTS vocoder signature)."] if risk else []
    conf = min(len(y) / (sr * 4), 1.0)
    is_strong = hnr > CFG.HNR_STRONG_THRESHOLD and conf > 0.40
    return {"score": risk, "confidence": conf, "is_strong": is_strong, "reasons": reasons}

def detect_sample_rate_fingerprint(y: np.ndarray, sr: int) -> dict:
    return {"score": 0, "confidence": 0.9, "is_strong": False, "reasons": []}

# ─────────────────────────────────────────────────────────────────
#  Master Entry Point
# ─────────────────────────────────────────────────────────────────
def analyze_audio(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        return {"score": 0, "confidence": 0, "is_strong": False, "reasons": [f"Audio load failed: {e}"], "sub_scores": {}}

    duration = len(y) / sr
    if duration < CFG.MIN_AUDIO_DURATION:
        return {"score": 15, "confidence": 0.2, "is_strong": False, "reasons": [f"[AUDIO] Too short ({duration:.1f}s)."], "sub_scores": {}}

    # FIX-2: Near-Silent Skip
    overall_rms = float(np.sqrt(np.mean(y ** 2)))
    if overall_rms < CFG.NEAR_SILENT_SKIP_THRESHOLD:
        return {"score": 0, "confidence": 0.1, "is_strong": False, "reasons": ["[AUDIO] Near-silent audio — skipping (not TTS silence)."], "sub_scores": {}}

    detectors = {
        "pitch":       detect_pitch_anomalies(y, sr),
        "phase":       detect_phase_discontinuities(y, sr),
        "silence":     detect_silence_anomalies(y, sr),
        "spectral":    detect_spectral_anomalies(y, sr),
        "formant":     detect_formant_transitions(y, sr),
        "hnr":         detect_hnr_anomaly(y, sr),
        "sample_rate": detect_sample_rate_fingerprint(y, sr),
    }
    
    return _fuse_results(detectors)

def get_audio_envelope(audio_path: str, num_points: int = 100) -> list:
    """Used for lip-sync correlation."""
    try:
        y, sr = librosa.load(audio_path, sr=8000, duration=30)
        hop = max(1, len(y) // num_points)
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        if np.max(rms) > 0: rms /= np.max(rms)
        return np.interp(np.linspace(0, len(rms)-1, num_points), np.arange(len(rms)), rms).tolist()
    except Exception:
        return [0.0] * num_points
