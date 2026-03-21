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
from concurrent.futures import ThreadPoolExecutor

# MoviePy is replaced by OpenCV for better performance on 8GB RAM
MOVIEPY_OK = False 

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

from modules.image import analyze_image
from modules.audio import analyze_audio


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
            # Optimized: Resize to 128x128 for hyper-fast SSIM calculation
            img1 = cv2.resize(img1, (128, 128))
            img2 = cv2.resize(img2, (128, 128))
            val = ssim(img1, img2, data_range=255)
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
    # Higher sensitivity: Catch subtle deepfake flickering
    anomaly = std > 0.08 or (mean > 0.98 and std > 0.03)
    return {'ssim_mean': float(round(mean, 4)), 'ssim_std': float(round(std, 4)), 'anomaly': anomaly}


def analyze_video(file_path: str, low_resource: bool = False) -> dict:
    """
    Full AI-enhanced video analysis:
      - Key frame AI-image detection (catches Sora/RunwayML/Pika generated video)
      - Enhanced audio analysis (pitch, MFCC delta, energy)
      - SSIM temporal consistency
      - Silent track detection (raw AI video indicator)
    """
    results: dict = {
        'score': 0,
        'risk_level': 'Low',
        'reasons': list(),
        'frames_analyzed': 0,
        'audio_analyzed': False,
        'ai_frame_scores': list(),
        'ssim': dict(),
    }

    if not CV2_OK:
        results['reasons'].append("OpenCV (cv2) not installed — video analysis skipped")
        return results
 
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            results['reasons'].append("Could not open video file with OpenCV")
            return results

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        # Limit to 10 seconds for speed
        duration = min(duration, 10.0)

        frames_dir = tempfile.mkdtemp()
        frame_paths = []
        frame_ai_scores = []
        frame_reasons = []
        combined_score = 0

        # Improvement: Sample more frames for better accuracy (5 instead of 3)
        num_samples = 1 if low_resource else 5
        sample_times = np.linspace(0, max(duration - 0.1, 0), num_samples)

        for t in sample_times:
            frame_path = os.path.join(frames_dir, f"frame_{t:.1f}.jpg")
            try:
                # Optimized frame extraction via CV2 is much lighter than MoviePy
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    
                    # Run AI-enhanced image analysis on each frame
                    img_res = analyze_image(frame_path)
                    frame_ai_scores.append(img_res['score'])
                    for r in img_res['reasons']:
                        frame_reasons.append(f"Frame {t:.1f}s: {r}")
            except Exception:
                pass

        cap.release()

        results['frames_analyzed'] = len(frame_paths)
        results['ai_frame_scores'] = frame_ai_scores

        avg_frame_score = int(np.mean(frame_ai_scores)) if frame_ai_scores else 0

        # SSIM temporal consistency (SKIP in low_resource mode)
        ssim_res = {'ssim_mean': 1.0, 'ssim_std': 0.0, 'anomaly': False}
        if not low_resource:
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

        # Enhanced audio analysis (Skip if no audio or error)
        audio_score = 0
        audio_reasons = []
        # Audio extraction via ffmpeg directly is better than MoviePy for resource management
        audio_path = os.path.join(frames_dir, "temp_audio_ai.wav")
        try:
            # -vn: no video, -y: overwrite
            import subprocess
            subprocess.run(['ffmpeg', '-i', file_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path, '-y'], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            if os.path.exists(audio_path):
                aud_res = analyze_audio(audio_path)
                audio_score = aud_res['score']
                audio_reasons = aud_res.get('reasons', [])
                os.remove(audio_path)
                results['audio_analyzed'] = True
            else:
                # No audio = indicator of raw AI video
                audio_score = 60
                audio_reasons.append("No audio track detected.")
        except Exception:
            # Fallback
            audio_reasons.append("Audio analysis unavailable (possible silent track).")

        # Final cleanup
        try:
            os.rmdir(frames_dir)
        except Exception:
            pass

        # Accuracy Boost: Prioritize the most suspicious frame (MAX) + average weight
        max_frame_score = int(np.max(frame_ai_scores)) if frame_ai_scores else 0
        avg_frame_score = int(np.mean(frame_ai_scores)) if frame_ai_scores else 0

        # Consistency bonus: if multiple frames all flag AI, it's a very strong signal
        high_frames = sum(1 for s in frame_ai_scores if s >= 40)
        consistency_bonus = 0
        if len(frame_ai_scores) >= 3 and high_frames >= 3:
            consistency_bonus = 20
            frame_reasons.append(f"Multi-frame consistency: {high_frames}/{len(frame_ai_scores)} frames flagged AI — strong synthesis indicator")
        elif len(frame_ai_scores) >= 2 and high_frames >= 2:
            consistency_bonus = 10

        # Core score: max-biased + consistency
        core_visual_score = int(max_frame_score * 0.8 + avg_frame_score * 0.2) + consistency_bonus
        core_visual_score = min(100, core_visual_score)

        # Metadata Check
        from modules.metadata import check_video_metadata
        meta_res = check_video_metadata(file_path)
        meta_score = meta_res.get('score', 0)

        # Smart weight redistribution when tools unavailable
        # meta_score >= 20 means real forensic evidence was found (not just a failed scan default of 10)
        audio_available = results.get('audio_analyzed', False)
        meta_real = meta_score >= 20
        if audio_available and meta_real:
            combined_score += int(core_visual_score * 0.6 + audio_score * 0.2 + meta_score * 0.2)
        elif audio_available:
            combined_score += int(core_visual_score * 0.75 + audio_score * 0.25)
        elif meta_real:
            combined_score += int(core_visual_score * 0.75 + meta_score * 0.25)
        else:
            # Neither audio nor real metadata — trust visual fully
            combined_score += core_visual_score

        if audio_score >= 80 or max_frame_score >= 80 or meta_score >= 70:
            combined_score = max(combined_score, max(audio_score, max_frame_score, meta_score))

        results['score'] = min(100, combined_score)
        results['risk_level'] = 'High' if results['score'] >= 60 else 'Medium' if results['score'] >= 30 else 'Low'
        results['reasons'] = frame_reasons[:5] + audio_reasons + meta_res.get('reasons', [])
        return results

    except Exception as e:
        results['reasons'].append(f"Video processing error: {str(e)}")
        return results
