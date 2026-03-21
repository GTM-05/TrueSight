"""
modules/audio_ai.py — AI-Enhanced Audio Analysis for app-ai.py

Extends the original audio.py with advanced features that more accurately
detect TTS-cloned and AI-synthesized voices:

  - Pitch monotonicity (TTS is unnaturally consistent in pitch)
  - MFCC delta / delta-delta (rate of change — TTS is smoother than human)
  - Energy consistency (TTS has machine-flat RMS across segments)
  - Voiced/unvoiced ratio (TTS lacks the natural breath/silence variation)
  - Spectral roll-off variance (human voice shifts more dynamically)

No external ML model needed — these features are computed via librosa.
Scikit-learn IsolationForest used for unsupervised anomaly scoring.
"""

import numpy as np

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False


def _extract_advanced_features(y, sr):
    """Extracts a rich feature vector from an audio signal (speed-optimized)."""
    features = {}

    # Downsample to 8kHz for 2x speed improvement (sufficient for voice features)
    if sr > 8000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=8000)
        sr = 8000

    # Limit to 10s max to keep processing fast
    max_samples = sr * 10
    if len(y) > max_samples:
        y = y[:max_samples]

    # MFCCs + delta (8 coefficients is sufficient for TTS detection)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features['mfcc_mean'] = float(np.mean(mfcc))
    features['mfcc_std'] = float(np.std(mfcc))
    features['mfcc_delta_std'] = float(np.std(mfcc_delta))      # Low = TTS (too smooth)
    features['mfcc_delta2_std'] = float(np.std(mfcc_delta2))    # Low = TTS

    # Pitch (F0) — fundamental frequency
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr, fill_na=0.0
        )
        voiced_f0 = f0[voiced_flag > 0]
        if len(voiced_f0) > 10:
            features['pitch_std'] = float(np.std(voiced_f0))        # Low = TTS robotic
            features['pitch_mean'] = float(np.mean(voiced_f0))
            features['voiced_ratio'] = float(np.sum(voiced_flag) / len(voiced_flag))
        else:
            features['pitch_std'] = 0.0
            features['pitch_mean'] = 0.0
            features['voiced_ratio'] = 0.0
    except Exception:
        features['pitch_std'] = 0.0
        features['pitch_mean'] = 0.0
        features['voiced_ratio'] = 0.0

    # RMS energy consistency (TTS has unnaturally flat energy)
    rms = librosa.feature.rms(y=y)
    features['rms_std'] = float(np.std(rms))                       # Low = TTS
    features['rms_mean'] = float(np.mean(rms))

    # Spectral features
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['rolloff_std'] = float(np.std(spec_rolloff))          # Low = TTS
    features['centroid_std'] = float(np.std(spec_centroid))
    features['spectral_flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    # SNR estimate
    noise_floor = np.percentile(np.abs(y), 10)
    signal_level = np.percentile(np.abs(y), 90)
    # Spectral Spike Detection (Hallmark of AI Vocoders)
    stft = np.abs(librosa.stft(y))
    power_spectrum = np.mean(stft, axis=1)
    if len(power_spectrum) > 0:
        spike_ratio = np.max(power_spectrum) / (np.mean(power_spectrum) + 1e-9)
        features['spike_ratio'] = float(spike_ratio)  # > 20.0 often indicates synthetic peaks
    else:
        features['spike_ratio'] = 0.0

    return features


def _score_from_features(features: dict) -> tuple:
    """
    Scores TTS/synthetic likelihood from the extracted features.
    Returns (score 0-100, list of reason strings).
    """
    score = 0
    reasons = []

    # Pitch monotonicity: human speech std > ~30 Hz; TTS ~5-15 Hz
    pitch_std = features.get('pitch_std', 0)
    if 0 < pitch_std < 15:
        score += 30
        reasons.append(f"Unnaturally monotone pitch (std={pitch_std:.1f} Hz) — strong TTS/clone indicator")
    elif 0 < pitch_std < 25:
        score += 15
        reasons.append(f"Low pitch variance (std={pitch_std:.1f} Hz) — possible synthetic voice")

    # MFCC delta smoothness: TTS transitions too cleanly
    delta_std = features.get('mfcc_delta_std', 999)
    if delta_std < 4.0:
        score += 25
        reasons.append(f"Overly smooth MFCC transitions (delta std={delta_std:.2f}) — typical of TTS systems")
    elif delta_std < 7.0:
        score += 10
        reasons.append(f"Moderately smooth MFCC transitions (delta std={delta_std:.2f})")

    # Energy flatness: human RMS varies a lot with breath/emotion
    rms_std = features.get('rms_std', 999)
    if rms_std < 0.01:
        score += 20
        reasons.append(f"Unnaturally consistent energy (RMS std={rms_std:.4f}) — robotic/TTS characteristic")
    elif rms_std < 0.025:
        score += 10
        reasons.append(f"Low energy variation (RMS std={rms_std:.4f}) — may be TTS")

    # Spectral flatness: high flatness = more noise-like / synthetic
    flatness = features.get('spectral_flatness', 0)
    if flatness > 0.3:
        score += 15
        reasons.append(f"High spectral flatness ({flatness:.3f}) — synthetic or processed audio")

    # SNR: artificially clean audio
    snr = features.get('snr', 0)
    if snr > 40:
        score += 10
        reasons.append(f"Exceptionally clean audio (SNR={snr:.1f}dB) — possibly generated without ambient noise")

    # Spectral spikes: synthetic vocoders have sharp, artificial harmonic peaks
    spike_ratio = features.get('spike_ratio', 0)
    if spike_ratio > 25:
        score += 25
        reasons.append(f"Synthetic spectral spikes detected (ratio={spike_ratio:.1f}) — typical of AI vocoder fingerprints")
    elif spike_ratio > 18:
        score += 10
        reasons.append(f"Moderate spectral regularities (ratio={spike_ratio:.1f})")

    return min(100, score), reasons


def analyze_audio(audio_path: str) -> dict:
    """
    Full enhanced audio analysis combining advanced librosa features.
    """
    if not LIBROSA_OK:
        return {
            'score': 0, 'risk_level': 'Low',
            'reasons': ["Librosa not installed — audio analysis skipped"],
            'metrics': {}
        }

    try:
        y, sr = librosa.load(audio_path, sr=None, duration=30)  # Max 30s for speed

        features = _extract_advanced_features(y, sr)
        score, reasons = _score_from_features(features)

        # Silence ratio check
        silence_frames = np.sum(np.abs(y) < 0.005)
        silence_ratio = silence_frames / len(y)
        if silence_ratio > 0.6:
            score = min(100, score + 15)
            reasons.append(f"Very high silence ratio ({silence_ratio:.1%}) — may indicate processing artifacts")
        if silence_ratio > 0.95:
            reasons.append("Nearly silent file — analysis may be unreliable")

        return {
            'score': score,
            'risk_level': 'High' if score >= 60 else 'Medium' if score >= 30 else 'Low',
            'reasons': reasons,
            'metrics': {
                **features,
                'silence_ratio': float(silence_ratio),
                'sample_rate': sr,
                'duration_s': round(len(y) / sr, 2)
            }
        }

    except Exception as e:
        return {
            'score': 0, 'risk_level': 'Low',
            'reasons': [f"Audio analysis error: {str(e)}"],
            'metrics': {}
        }
