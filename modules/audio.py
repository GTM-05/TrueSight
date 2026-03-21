import librosa
import numpy as np
import warnings

# Ignore librosa warnings for short audio files
warnings.filterwarnings('ignore')

def analyze_audio(audio_path):
    """
    Extracts audio features and applies heuristic rules to detect 
    synthetic/deepfake audio properties (e.g. flat pitch, missing frequencies).
    """
    score = 0
    reasons = []
    
    try:
        # Load audio (downsample to 16kHz for speed)
        y, sr = librosa.load(audio_path, sr=16000)
        
        if len(y) == 0:
            return {'score': 0, 'reasons': ['Empty audio file'], 'risk_level': 'Low'}
            
        # 1. MFCC Analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.var(mfccs, axis=1)
        # Deepfakes sometimes struggle with variance in higher MFCC bands
        high_band_var = np.mean(mfcc_var[8:])
        
        if high_band_var < 50:
            score += 25
            reasons.append(f"Low variance in high-frequency MFCCs ({high_band_var:.2f}) - typical in synthetic/generated audio")
            
        # 2. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        mean_flatness = np.mean(flatness)
        if mean_flatness < 0.001:
            score += 20
            reasons.append(f"Extremely low spectral flatness ({mean_flatness:.5f}) - signifies highly pure tones. Voice usually has natural noise characteristics.")
            
        # 3. Pitch Tracking (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            if magnitudes[index, i] > 2.0:
                pitch_vals.append(pitches[index, i])
                
        if len(pitch_vals) > 0:
            pitch_std = np.std(pitch_vals)
            # Human speech has natural pitch variations. Robotic TTS or poor deepfakes might be too flat
            if pitch_std < 15.0:
                score += 30
                reasons.append(f"Unnaturally flat pitch variance ({pitch_std:.2f} Hz) - sounds robotic or like poor Text-to-Speech")
            # Or wildly erratic
            elif pitch_std > 300.0:
                score += 20
                reasons.append(f"Erratic pitch variance ({pitch_std:.2f} Hz) - unnatural frequency leaps not common in normal human speech")
                
        # 4. Silence analysis
        non_mute_intervals = librosa.effects.split(y, top_db=30)
        total_duration = len(y)
        active_duration = sum([i[1] - i[0] for i in non_mute_intervals])
        silence_ratio = 1.0 - (active_duration / total_duration)
        
        if silence_ratio > 0.6:
            score += 10
            reasons.append(f"High silence ratio ({silence_ratio:.0%}) - file is mostly empty or consists of highly isolated words")
            
        metrics = {
            'high_band_mfcc_var': float(high_band_var),
            'spectral_flatness': float(mean_flatness),
            'silence_ratio': float(silence_ratio),
            'pitch_std': float(pitch_std) if 'pitch_std' in locals() else 0.0
        }
        
    except Exception as e:
        score += 10
        reasons.append(f"Error processing audio: {str(e)}")
        metrics = {}
        
    final_score = min(100, score)
    return {
        'score': final_score,
        'reasons': reasons,
        'metrics': metrics,
        'risk_level': 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'
    }
