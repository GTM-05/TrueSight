"""
TrueSight — Strong Audio Forensics Module
modules/audio.py

Multi-layer vocal forensics: pitch, silence, spectral, formant, phase, AV-sync.
All analysis is offline, Librosa/NumPy/SciPy only — no external model required.
"""

import numpy as np
import librosa
import cv2
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from scipy.stats import kurtosis, skew
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────
@dataclass
class AudioForensicConfig:
    # Pitch / F0
    F0_MIN_HZ: float = 60.0
    F0_MAX_HZ: float = 400.0
    JITTER_SYNTHETIC_THRESHOLD: float = 0.012   # Below = robotic
    SHIMMER_SYNTHETIC_THRESHOLD: float = 0.04   # Below = robotic
    PITCH_MONOTONE_VAR_THRESHOLD: float = 15.0  # Hz std — very flat = TTS

    # Silence / noise floor
    SILENCE_RMS_THRESHOLD: float = 0.0018       # Digital-zero silence floor
    SILENCE_NOISE_STD_MAX: float = 0.0004       # Real mics always have noise
    DIGITAL_SILENCE_RATIO_MIN: float = 0.08     # > 8 % digital-zero = suspicious

    # Spectral
    SPECTRAL_FLATNESS_TTS_MIN: float = 0.13     # Vocoders: flat spectra
    SPECTRAL_ROLLOFF_AI_MIN: float = 0.78       # AI audio rolls off less naturally

    # MFCC / Formant transitions
    MFCC_ABRUPT_PERCENTILE: float = 97.0
    MFCC_ABRUPT_RATIO_MAX: float = 0.038

    # Phase / Edit cuts
    PHASE_DISC_WINDOW: int = 2048
    PHASE_DISC_THRESHOLD: float = 1.4           # Radians — sudden wrap = splice

    # Sample rate fingerprint (TTS defaults)
    TTS_SAMPLE_RATES: tuple = (22050, 24000, 16000)
    HUMAN_SAMPLE_RATES: tuple = (44100, 48000, 96000)

    # Harmonic structure
    HNR_SYNTHETIC_MIN: float = 22.0             # TTS > 22 dB HNR (too clean)

    # Minimum audio length (seconds) for reliable analysis
    MIN_AUDIO_DURATION: float = 2.0

    # Risk weights per detector
    WEIGHTS: dict = field(default_factory=lambda: {
        "pitch_jitter":        30,
        "pitch_shimmer":       20,
        "pitch_monotone":      20,
        "digital_silence":     25,
        "spectral_flatness":   20,
        "spectral_rolloff":    15,
        "formant_transitions": 20,
        "phase_discontinuity": 35,
        "sample_rate_flag":    10,
        "hnr_too_clean":       20,
        "mfcc_delta_flatness": 15,
    })


CFG = AudioForensicConfig()


# ─────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────
def _bandpass(signal: np.ndarray, lo: float, hi: float, sr: int,
              order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


def _hnr(y: np.ndarray, sr: int, frame_len: int = 2048,
         hop: int = 512) -> float:
    """Harmonic-to-Noise Ratio via autocorrelation."""
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
    hnr_vals = []
    for f in frames.T:
        acf = np.correlate(f, f, mode="full")[frame_len - 1:]
        if acf[0] < 1e-10:
            continue
        acf /= acf[0]
        peaks, _ = find_peaks(acf[1:], height=0.1)
        if len(peaks) == 0:
            continue
        r0 = acf[0]
        r1 = acf[peaks[0] + 1]
        if r1 >= r0:
            continue
        hnr_vals.append(10 * np.log10(r1 / (r0 - r1 + 1e-10)))
    return float(np.median(hnr_vals)) if hnr_vals else 0.0


def _phase_discontinuities(y: np.ndarray, sr: int,
                            win: int = 2048, hop: int = 512) -> list[float]:
    """Detect abrupt phase wraps — signatures of audio edit splices."""
    stft = librosa.stft(y, n_fft=win, hop_length=hop)
    phase = np.angle(stft)
    phase_diff = np.diff(phase, axis=1)
    # Instantaneous frequency deviation
    ifd = np.abs(phase_diff - np.median(phase_diff, axis=1, keepdims=True))
    # Mean IFD per frame
    return list(np.mean(ifd, axis=0))


# ─────────────────────────────────────────────────────────────────
#  Individual detectors
# ─────────────────────────────────────────────────────────────────

def detect_pitch_anomalies(y: np.ndarray, sr: int) -> dict:
    """
    F0 jitter, shimmer, and monotone detection.
    Human speech: jitter > 1.2 %, shimmer > 4 %, std(F0) > 15 Hz.
    TTS / VC: suspiciously stable and clean.
    """
    risk = 0
    reasons = []

    f0 = librosa.yin(y, fmin=CFG.F0_MIN_HZ, fmax=CFG.F0_MAX_HZ, sr=sr)
    voiced = f0[f0 > CFG.F0_MIN_HZ]

    if len(voiced) < 20:
        return {"score": 0, "confidence": 0.1, "reasons": ["Insufficient voiced frames for pitch analysis."]}

    # ── Jitter (cycle-to-cycle pitch variation) ──────────────────
    diffs = np.abs(np.diff(voiced))
    jitter = float(np.mean(diffs) / (np.mean(voiced) + 1e-9))
    if jitter < CFG.JITTER_SYNTHETIC_THRESHOLD:
        risk += CFG.WEIGHTS["pitch_jitter"]
        reasons.append(
            f"[PITCH] Robotic pitch stability — jitter={jitter:.4f} "
            f"(threshold < {CFG.JITTER_SYNTHETIC_THRESHOLD}). "
            f"Human voices show micro-variations > 1.2%. TTS/VC pipelines produce near-zero jitter."
        )

    # ── Shimmer (amplitude envelope variation per F0 period) ────
    rms_frames = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    if len(rms_frames) > 10:
        amp_diffs = np.abs(np.diff(rms_frames))
        shimmer = float(np.mean(amp_diffs) / (np.mean(rms_frames) + 1e-9))
        if shimmer < CFG.SHIMMER_SYNTHETIC_THRESHOLD:
            risk += CFG.WEIGHTS["pitch_shimmer"]
            reasons.append(
                f"[PITCH] Unnatural amplitude stability — shimmer={shimmer:.4f} "
                f"(threshold < {CFG.SHIMMER_SYNTHETIC_THRESHOLD}). "
                f"Real voices have breath-driven amplitude fluctuations."
            )

    # ── Monotone detection ───────────────────────────────────────
    pitch_std = float(np.std(voiced))
    if pitch_std < CFG.PITCH_MONOTONE_VAR_THRESHOLD:
        risk += CFG.WEIGHTS["pitch_monotone"]
        reasons.append(
            f"[PITCH] Extremely monotone delivery — F0 std={pitch_std:.1f} Hz "
            f"(threshold < {CFG.PITCH_MONOTONE_VAR_THRESHOLD} Hz). "
            f"Human prosody naturally varies; TTS models default to flat intonation."
        )

    confidence = min(len(voiced) / 100, 1.0)
    return {
        "score": min(risk, 100),
        "confidence": confidence,
        "is_strong": risk >= 45,
        "reasons": reasons,
        "debug": {"jitter": jitter, "pitch_std": pitch_std}
    }


def detect_silence_anomalies(y: np.ndarray, sr: int) -> dict:
    """
    Digital-zero silence = TTS. Real mics always have a noise floor.
    """
    risk = 0
    reasons = []

    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    silence_mask = rms < CFG.SILENCE_RMS_THRESHOLD

    if silence_mask.sum() == 0:
        return {"score": 0, "confidence": 0.6, "reasons": ["No silence segments found."]}

    silence_ratio = float(silence_mask.sum() / len(rms))
    silence_noise = rms[silence_mask]
    silence_std = float(np.std(silence_noise))

    if silence_std < CFG.SILENCE_NOISE_STD_MAX and silence_ratio > CFG.DIGITAL_SILENCE_RATIO_MIN:
        risk += CFG.WEIGHTS["digital_silence"]
        reasons.append(
            f"[SILENCE] Digital-zero silence detected — ratio={silence_ratio:.1%}, "
            f"noise_std={silence_std:.5f} (threshold < {CFG.SILENCE_NOISE_STD_MAX}). "
            f"Real microphones always capture ambient noise floor. "
            f"TTS renders silence as exact zeros."
        )

    confidence = min(silence_ratio * 5, 1.0) if silence_ratio > 0 else 0.3
    return {
        "score": min(risk, 100),
        "confidence": confidence,
        "is_strong": risk >= 25,
        "reasons": reasons
    }


def detect_spectral_anomalies(y: np.ndarray, sr: int) -> dict:
    """
    Spectral flatness and rolloff — vocoder / neural codec fingerprints.
    """
    risk = 0
    reasons = []

    # ── Spectral Flatness ────────────────────────────────────────
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    if flatness > CFG.SPECTRAL_FLATNESS_TTS_MIN:
        risk += CFG.WEIGHTS["spectral_flatness"]
        reasons.append(
            f"[SPECTRAL] High spectral flatness={flatness:.4f} "
            f"(threshold > {CFG.SPECTRAL_FLATNESS_TTS_MIN}). "
            f"Vocoders and neural codecs (EnCodec, HiFi-GAN) produce "
            f"unnaturally uniform spectral energy distribution."
        )

    # ── Spectral Rolloff ─────────────────────────────────────────
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_norm = float(np.mean(rolloff) / (sr / 2))
    if rolloff_norm > CFG.SPECTRAL_ROLLOFF_AI_MIN:
        risk += CFG.WEIGHTS["spectral_rolloff"]
        reasons.append(
            f"[SPECTRAL] Abnormal spectral rolloff={rolloff_norm:.3f} "
            f"(normalized, threshold > {CFG.SPECTRAL_ROLLOFF_AI_MIN}). "
            f"Indicates energy pushed to high frequencies — artifact of neural vocoders."
        )

    # ── MFCC Delta Flatness (frozen timbre) ──────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = np.diff(mfcc, axis=1)
    delta_var = float(np.mean(np.var(mfcc_delta, axis=1)))
    if delta_var < 1.5:
        risk += CFG.WEIGHTS["mfcc_delta_flatness"]
        reasons.append(
            f"[SPECTRAL] Frozen timbre — MFCC delta variance={delta_var:.3f} "
            f"(threshold < 1.5). Voice cloning models produce static timbre "
            f"with minimal temporal MFCC variation."
        )

    confidence = min(len(y) / (sr * 3), 1.0)
    return {
        "score": min(risk, 100),
        "confidence": confidence,
        "is_strong": risk >= 35,
        "reasons": reasons,
        "debug": {"flatness": flatness, "delta_var": delta_var}
    }


def detect_formant_transitions(y: np.ndarray, sr: int) -> dict:
    """
    Abrupt MFCC jumps = voice cloning splice artifacts.
    """
    risk = 0
    reasons = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = np.abs(np.diff(mfcc, axis=1))
    threshold = np.percentile(delta, CFG.MFCC_ABRUPT_PERCENTILE)
    abrupt = np.sum(delta > threshold)
    ratio = float(abrupt / delta.size)

    if ratio > CFG.MFCC_ABRUPT_RATIO_MAX:
        risk += CFG.WEIGHTS["formant_transitions"]
        reasons.append(
            f"[FORMANT] Abrupt formant transitions — ratio={ratio:.4f} "
            f"(threshold > {CFG.MFCC_ABRUPT_RATIO_MAX}). "
            f"Voice cloning systems produce sharp coarticulation artifacts "
            f"at phoneme boundaries due to segment stitching."
        )

    confidence = min(mfcc.shape[1] / 50, 1.0)
    return {
        "score": min(risk, 100),
        "confidence": confidence,
        "is_strong": risk >= 20,
        "reasons": reasons
    }


def detect_phase_discontinuities(y: np.ndarray, sr: int) -> dict:
    """
    Phase wraps at edit points — strongest signal for audio splicing.
    """
    risk = 0
    reasons = []

    ifd_series = _phase_discontinuities(y, sr, win=CFG.PHASE_DISC_WINDOW)
    if len(ifd_series) < 5:
        return {"score": 0, "confidence": 0.1, "reasons": ["Insufficient frames for phase analysis."]}

    ifd_arr = np.array(ifd_series)
    spike_mask = ifd_arr > CFG.PHASE_DISC_THRESHOLD
    spike_count = int(spike_mask.sum())
    spike_ratio = float(spike_count / len(ifd_arr))

    if spike_count >= 2:
        risk += CFG.WEIGHTS["phase_discontinuity"]
        reasons.append(
            f"[PHASE] {spike_count} phase discontinuities detected "
            f"(ratio={spike_ratio:.3f}, threshold > {CFG.PHASE_DISC_THRESHOLD} rad). "
            f"Sudden instantaneous frequency jumps are a definitive marker of "
            f"audio splice editing used in voice deepfake assembly."
        )

    confidence = min(len(ifd_arr) / 30, 1.0)
    return {
        "score": min(risk, 100),
        "confidence": confidence,
        "is_strong": spike_count >= 3,
        "reasons": reasons,
        "debug": {"spike_count": spike_count}
    }


def detect_hnr_anomaly(y: np.ndarray, sr: int) -> dict:
    """
    HNR too clean = TTS. Real voices have noise + breathiness.
    """
    risk = 0
    reasons = []

    hnr = _hnr(y, sr)
    if hnr > CFG.HNR_SYNTHETIC_MIN:
        risk += CFG.WEIGHTS["hnr_too_clean"]
        reasons.append(
            f"[HNR] Suspiciously clean harmonic structure — HNR={hnr:.1f} dB "
            f"(threshold > {CFG.HNR_SYNTHETIC_MIN} dB). "
            f"Human voices are naturally noisy (breath, aspiration). "
            f"Neural vocoders generate near-perfect harmonics."
        )

    confidence = min(len(y) / (sr * 4), 1.0)
    return {
        "score": min(risk, 100),
        "confidence": confidence,
        "is_strong": hnr > 28.0,
        "reasons": reasons,
        "debug": {"hnr_db": hnr}
    }


def detect_sample_rate_fingerprint(y: np.ndarray, sr: int) -> dict:
    """
    TTS pipelines default to 22050 / 24000 Hz.
    Human recordings are typically 44100 / 48000 Hz.
    """
    risk = 0
    reasons = []

    duration = len(y) / sr
    if sr in CFG.TTS_SAMPLE_RATES and duration > 2.5:
        risk += CFG.WEIGHTS["sample_rate_flag"]
        reasons.append(
            f"[METADATA] Suspicious sample rate {sr} Hz — "
            f"common default in TTS pipelines (Tacotron2, VITS, Bark). "
            f"Human recordings are typically 44.1 kHz or 48 kHz."
        )

    return {
        "score": risk,
        "confidence": 0.9,
        "is_strong": False,  # Alone it's weak — combine with others
        "reasons": reasons
    }


# ─────────────────────────────────────────────────────────────────
#  Master entry point
# ─────────────────────────────────────────────────────────────────
def analyze_audio(audio_path: str) -> dict:
    """
    Run full audio forensics pipeline.

    Returns:
        score       : 0–100 overall risk
        confidence  : 0–1 how much data backed the analysis
        is_strong   : True if at least one definitive anchor fired
        reasons     : List of human-readable forensic findings
        sub_scores  : Per-detector breakdown
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        return {"score": 0, "confidence": 0, "is_strong": False,
                "reasons": [f"Audio load failed: {e}"], "sub_scores": {}}

    duration = len(y) / sr
    if duration < CFG.MIN_AUDIO_DURATION:
        return {"score": 15, "confidence": 0.2, "is_strong": False,
                "reasons": [f"Audio too short ({duration:.1f}s) for reliable forensics."],
                "sub_scores": {}}

    # ── Run all detectors ────────────────────────────────────────
    detectors = {
        "pitch":       detect_pitch_anomalies(y, sr),
        "silence":     detect_silence_anomalies(y, sr),
        "spectral":    detect_spectral_anomalies(y, sr),
        "formant":     detect_formant_transitions(y, sr),
        "phase":       detect_phase_discontinuities(y, sr),
        "hnr":         detect_hnr_anomaly(y, sr),
        "sample_rate": detect_sample_rate_fingerprint(y, sr),
    }

    all_reasons = []
    for d in detectors.values():
        all_reasons.extend(d.get("reasons", []))

    # ── Confidence-weighted fusion ───────────────────────────────
    # Strong anchors dominate; weak signals are gated by confidence
    strong = [d for d in detectors.values()
              if d.get("is_strong") and d.get("confidence", 0) > 0.5]

    if strong:
        final_score = max(d["score"] for d in strong)
        # Accumulate corroborating weak signals (capped contribution)
        weak_boost = sum(
            d["score"] * d.get("confidence", 0.5) * 0.15
            for d in detectors.values()
            if not d.get("is_strong") and d.get("score", 0) > 10
        )
        final_score = min(final_score + weak_boost, 100)
    else:
        # No strong anchor — weighted average of confident signals
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

    overall_confidence = float(np.mean([d.get("confidence", 0.5)
                                         for d in detectors.values()]))
    has_strong = len(strong) > 0

    return {
        "score":      round(min(final_score, 100), 1),
        "confidence": round(overall_confidence, 3),
        "is_strong":  has_strong,
        "reasons":    all_reasons,
        "sub_scores": {k: {"score": v["score"], "confidence": v.get("confidence", 0)}
                       for k, v in detectors.items()},
        "duration_s": round(duration, 2),
        "sample_rate": sr,
    }

def get_audio_envelope(audio_path: str, num_points: int = 100) -> list:
    """
    Extracts a simplified RMS envelope from an audio file.
    Used for lip-sync correlation with video Mouth Aspect Ratio (MAR).
    """
    try:
        y, sr = librosa.load(audio_path, sr=8000, duration=30)
        # Calculate RMS with a window size that gives us ~num_points
        hop_length = max(1, len(y) // num_points)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        # Normalize to 0-1
        if np.max(rms) > 0:
            rms = rms / np.max(rms)
        # Interpolate/pad to exact num_points
        return np.interp(np.linspace(0, len(rms)-1, num_points), np.arange(len(rms)), rms).tolist()
    except Exception:
        return [0.0] * num_points
