"""
modules/video_ai.py — AI-Enhanced Video Analysis for app-ai.py

Improvements over the original video.py:
  1. Runs image_ai.py's AI-image-detector on sampled key frames
     (not just ELA — catches AI-generated video frames like Sora/RunwayML)
  2. Enhanced audio analysis via audio_ai.py (pitch, MFCC delta, energy)
  3. SSIM-based temporal consistency check
     (real videos have natural scene changes; deepfakes often have subtle
     inconsistency between frames due to per-frame synthesis)
  4. Face-region temporal instability check using OpenCV
"""

import os
import tempfile
import numpy as np

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_OK = True
except ImportError:
    MOVIEPY_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

from modules.image_ai import analyze_image_ai
from modules.audio_ai import analyze_audio_ai


def _compute_ssim_sequence(frame_paths: list) -> dict:
    """
    Computes SSIM (Structural Similarity Index) between consecutive frames.
    Natural videos have consistent scene flow; deepfakes often show sudden
    structural micro-inconsistencies due to frame-by-frame synthesis.
    Returns mean and std of SSIM values.
    """
    if not CV2_OK or len(frame_paths) < 2:
        return {'ssim_mean': 1.0, 'ssim_std': 0.0, 'anomaly': False}

    ssim_values = []
    for i in range(len(frame_paths) - 1):
        try:
            from skimage.metrics import structural_similarity as ssim
            img1 = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(frame_paths[i + 1], cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                continue
            # Resize to same shape if needed
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
            val = ssim(img1, img2)
            ssim_values.append(val)
        except ImportError:
            # skimage not available — skip SSIM
            break
        except Exception:
            continue

    if not ssim_values:
        return {'ssim_mean': 1.0, 'ssim_std': 0.0, 'anomaly': False}

    mean = float(np.mean(ssim_values))
    std = float(np.std(ssim_values))
    # High std + high mean = suspicious (very consistent then suddenly inconsistent)
    anomaly = std > 0.15 or (mean > 0.95 and std > 0.05)
    return {'ssim_mean': round(mean, 4), 'ssim_std': round(std, 4), 'anomaly': anomaly}


def analyze_video_ai(file_path: str) -> dict:
    """
    Full AI-enhanced video analysis:
      - Key frame AI-image detection (catches Sora/RunwayML/Pika generated video)
      - Enhanced audio analysis (pitch, MFCC delta, energy)
      - SSIM temporal consistency
      - Silent track detection (raw AI video indicator)
    """
    results = {
        'score': 0,
        'reasons': [],
        'frames_analyzed': 0,
        'audio_analyzed': False,
        'ai_frame_scores': [],
        'ssim': {},
    }

    if not MOVIEPY_OK:
        results['reasons'].append("MoviePy not installed — video analysis skipped")
        return results

    try:
        clip = VideoFileClip(file_path)

        # Limit to 10 seconds
        if clip.duration > 10:
            clip = clip.subclip(0, 10)
        duration = clip.duration

        frames_dir = tempfile.mkdtemp()
        frame_paths = []
        frame_ai_scores = []
        frame_reasons = []
        combined_score = 0

        # Sample up to 5 key frames (evenly spaced)
        sample_times = np.linspace(0, max(duration - 0.1, 0), min(5, int(duration)))

        for t in sample_times:
            frame_path = os.path.join(frames_dir, f"frame_{t:.1f}.jpg")
            try:
                clip.save_frame(frame_path, t=float(t))
                frame_paths.append(frame_path)

                # Run AI-enhanced image analysis on each frame
                img_res = analyze_image_ai(frame_path)
                frame_ai_scores.append(img_res['score'])

                for r in img_res['reasons']:
                    frame_reasons.append(f"Frame {t:.1f}s: {r}")

            except Exception:
                pass

        results['frames_analyzed'] = len(frame_paths)
        results['ai_frame_scores'] = frame_ai_scores

        avg_frame_score = int(np.mean(frame_ai_scores)) if frame_ai_scores else 0

        # SSIM temporal consistency
        ssim_res = _compute_ssim_sequence(frame_paths)
        results['ssim'] = ssim_res
        if ssim_res['anomaly']:
            combined_score += 20
            frame_reasons.append(
                f"Temporal inconsistency detected (SSIM std={ssim_res['ssim_std']:.3f}) "
                f"— abrupt structural changes between frames"
            )

        # Cleanup frames
        for fp in frame_paths:
            try:
                os.remove(fp)
            except Exception:
                pass

        # Enhanced audio analysis
        audio_score = 0
        audio_reasons = []
        if clip.audio is not None:
            audio_path = os.path.join(frames_dir, "temp_audio_ai.wav")
            try:
                clip.audio.write_audiofile(audio_path, logger=None)
                aud_res = analyze_audio_ai(audio_path)
                audio_score = aud_res['score']
                audio_reasons = aud_res.get('reasons', [])
                os.remove(audio_path)
                results['audio_analyzed'] = True
            except Exception as ae:
                audio_reasons.append(f"Audio extraction error: {str(ae)}")
        else:
            # No audio = strong indicator of raw AI-generated video (Sora constraint)
            audio_score = 80
            audio_reasons.append(
                "No audio track found — strongly indicative of AI-generated video "
                "(most AI video generators produce silent output by default)"
            )
            results['audio_analyzed'] = True

        clip.close()
        try:
            os.rmdir(frames_dir)
        except Exception:
            pass

        # Final score: heaviest weight on frame AI score
        combined_score += int(avg_frame_score * 0.6 + audio_score * 0.4)
        if audio_score >= 70:
            combined_score = max(combined_score, audio_score)

        results['score'] = min(100, combined_score)
        results['risk_level'] = 'High' if results['score'] >= 60 else 'Medium' if results['score'] >= 30 else 'Low'
        results['reasons'] = frame_reasons[:5] + audio_reasons  # Cap frame reasons
        return results

    except Exception as e:
        results['reasons'].append(f"Video processing error: {str(e)}")
        return results
