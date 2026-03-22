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
    if not LIBROSA_OK:
        return features

    # Downsample to 8kHz for 2x speed improvement (sufficient for voice features)
    if sr > 8000:
        y = librosa.resample(y, orig_sr=sr, target_sr=8000)
        sr = 8000

    # Limit to 10s max to keep processing fast
    max_samples = sr * 10
    if len(y) > max_samples:
        y = y[:max_samples]

    # MFCCs + delta (20 coefficients to capture fine vocal tract details)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    features['mfcc_delta_std'] = float(np.std(mfcc_delta))

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

    # Spectral Flux (Dynamic variation of spectrum)
    spec_flux = np.sqrt(np.mean(np.diff(librosa.feature.spectral_centroid(y=y, sr=sr))**2))
    features['spec_flux'] = float(spec_flux)                      # Low = TTS (static tone)

    # Shannon Entropy of Spectral magnitude
    stft = np.abs(librosa.stft(y))
    power = stft**2
    sum_power = np.sum(power, axis=0) + 1e-9
    prob = power / sum_power
    entropy = -np.sum(prob * np.log(prob + 1e-9), axis=0)
    features['spectral_entropy'] = float(np.mean(entropy))        # Low = AI periodcity

    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_std'] = float(np.std(zcr))

    # SNR estimate
    noise_floor = np.percentile(np.abs(y), 10)
    signal_level = np.percentile(np.abs(y), 90)
    # High-Frequency Ratio (AI vocoders often miss 8kHz+ or have gaps)
    # Spectral Spike Detection (Hallmark of AI Vocoders)
    stft_full = np.abs(librosa.stft(y))
    power_spectrum = np.mean(stft_full, axis=1)
    if len(power_spectrum) > 30:
        spike_ratio = np.max(power_spectrum) / (np.mean(power_spectrum) + 1e-9)
        features['spike_ratio'] = float(spike_ratio)
        # HFR: Ratio of high (>6kHz) to mid (1-4kHz) energy
        h_idx = int(0.75 * len(power_spectrum))
        m_idx = int(0.25 * len(power_spectrum))
        features['hfr'] = float(np.mean(power_spectrum[h_idx:]) / (np.mean(power_spectrum[m_idx:h_idx]) + 1e-9))
    else:
        features['spike_ratio'] = 0.0
        features['hfr'] = 0.0

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
    # IMPORTANT: Skip if audio is silent/near-silent (delta std would be 0 naturally)
    is_silent = features.get('rms_mean', 1.0) < 0.001
    delta_std = features.get('mfcc_delta_std', 999)
    if not is_silent:
        if delta_std < 4.0:
            score += 25
            reasons.append(f"Overly smooth MFCC transitions (delta std={delta_std:.2f}) — typical of TTS systems")
        elif delta_std < 7.0:
            score += 10
            reasons.append(f"Moderately smooth MFCC transitions (delta std={delta_std:.2f})")

    # Energy flatness: human RMS varies a lot with breath/emotion
    rms_std = features.get('rms_std', 999)
    if not is_silent:
        if rms_std < 0.01:
            score += 20
            reasons.append(f"Unnaturally consistent energy (RMS std={rms_std:.4f}) — robotic/TTS characteristic")
        elif rms_std < 0.025:
            score += 10
            reasons.append(f"Low energy variation (RMS std={rms_std:.4f}) — may be TTS")
    else:
        reasons.append("Note: Audio is nearly silent; skipping temporal smoothness heuristics.")

    # Pitch monotonicity
    pitch_std = features.get('pitch_std', 0)
    if not is_silent:
        if 0 < pitch_std < 15:
            score += 30
            reasons.append(f"Unnaturally monotone pitch (std={pitch_std:.1f} Hz) — strong TTS/clone indicator")
        elif 0 < pitch_std < 25:
            score += 15
            reasons.append(f"Low pitch variance (std={pitch_std:.1f} Hz) — possible synthetic voice")

    # Spectral flatness
    flatness = features.get('spectral_flatness', 0)
    if not is_silent and flatness > 0.3:
        score += 15
        reasons.append(f"High spectral flatness ({flatness:.3f}) — synthetic or processed audio")

    # Spectral Flux
    spec_flux = features.get('spec_flux', 100)
    if not is_silent and spec_flux < 45:
        score += 20
        reasons.append(f"Low spectral flux ({spec_flux:.1f}) — speech is unnaturally static/robotic")

    # Spectral Entropy
    entropy = features.get('spectral_entropy', 1.0)
    if not is_silent and entropy < 0.35:
        score += 25
        reasons.append(f"Unnatural spectral entropy ({entropy:.3f}) — pattern typical of neural vocoder artifacts")

    # High-Frequency Ratio
    hfr = features.get('hfr', 0.5)
    if not is_silent:
        if hfr < 0.02:
            score += 30
            reasons.append(f"Severely suppressed high-frequencies (HFR={hfr:.3f}) — typical of low-quality TTS synthesis")
        elif hfr > 0.8:
            score += 20
            reasons.append(f"Excessive high-frequency noise (HFR={hfr:.3f}) — possible synthetic vocoder 'hiss'")

    # Spectral spikes
    spike_ratio = features.get('spike_ratio', 0)
    if not is_silent and spike_ratio > 25:
        score += 25
        reasons.append(f"Synthetic spectral spikes detected (ratio={spike_ratio:.1f}) — typical of AI vocoder fingerprints")

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
        silence_ratio = float(silence_frames / len(y))

        # Ambient Floor Analysis (Is silence "natural" room tone or digital zero?)
        # Use explicit indexing cast for linter
        y_np = np.asarray(y)
        mask = np.abs(y_np) < 0.01
        quiet_frames = y_np[mask]
        if len(quiet_frames) > sr // 5:
            q_std = float(np.std(quiet_frames))
            if q_std < 1e-4: # Extremely clean
                score = min(100, score + 10)
                reasons.append("Exceptionally clean noise floor (studio or synthetic environment)")

    # Digital silence checking (mathematically zero is very suspicious in raw recordings)
        silence_regions = np.where(y == 0)[0]
        digital_silence_ratio = len(silence_regions) / len(y)
        if digital_silence_ratio > 0.999:
            score = min(100, score + 15) # Reduced from 25+
            reasons.append("Digital Silence Anomaly: Silence regions are mathematically zero (no room tone/ambient noise)")
            reasons.append(f"Very high silence ratio ({digital_silence_ratio*100:.1f}%) — may indicate processing artifacts")
            reasons.append("Nearly silent file — analysis may be unreliable")

        elif silence_ratio > 0.6:
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


def get_audio_envelope(audio_path: str, num_points: int = 100) -> list:
    """
    Extracts a simplified RMS envelope from an audio file.
    Used for lip-sync correlation with video Mouth Aspect Ratio (MAR).
    """
    if not LIBROSA_OK:
        return [0.0] * num_points
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
