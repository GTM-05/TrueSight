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
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("truesight.video")

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


def _compute_optical_flow_anomaly(frame_paths: list) -> dict:
    """
    Computes Optical Flow (Farneback) to detect motion warping.
    Morphed content (Deepfakes) often has 'unphysical' flow vectors
    where pixels slide unnaturally or show high-frequency direction jitter.
    """
    # TIER 3: Region-Aware Optical Flow (Face Focus)
    face_region = None
    try:
        from modules.image import _get_face_detector
        face_cascade = _get_face_detector()
        if face_cascade is not None:
            first_frame = cv2.imread(frame_paths[0])
            if first_frame is not None:
                gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_first, 1.1, 4)
                if len(faces) > 0:
                    face_region = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    except Exception:
        pass

    flow_mags = []
    try:
        prev_img = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
        if face_region is not None:
            x, y, w, h = face_region
            prev_img = prev_img[y:y+h, x:x+w]
        prev_img = cv2.resize(prev_img, (256, 256))
        
        for i in range(1, len(frame_paths)):
            curr_img = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
            if curr_img is None: continue
            if face_region is not None:
                x, y, w, h = face_region
                curr_img = curr_img[y:y+h, x:x+w]
            curr_img = cv2.resize(curr_img, (256, 256))
            
            # Fast Farneback Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev_img, curr_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mags.append(np.mean(mag))
            prev_img = curr_img
            
    except Exception:
        return {'flow_std': 0.0, 'anomaly': False}

    if not flow_mags:
        return {'flow_std': 0.0, 'anomaly': False}

    std = float(np.std(flow_mags))
    # std > 0.4 indicates sharp motion discontinuities on the face
    threshold = 0.4 if face_region is not None else 0.5
    anomaly = std > threshold
    return {
        'flow_std': round(std, 4), 
        'anomaly': anomaly, 
        'method': 'Face-Flow' if face_region is not None else 'Global-Flow'
    }


def _fast_heuristic_score(image_path: str) -> int:
    """
    Ultra-fast frame pre-scan (~0.2s) using FFT smoothness + resolution check only.
    Used in Step 1 of adaptive slicing to rank frames without running the heavy ViT model.
    Returns a score 0–55 (max 55 to distinguish from full ViT which can reach 80+).
    """
    score = 0
    try:
        from PIL import Image
        img = Image.open(image_path)
        w, h = img.size
        # Known AI output resolutions
        ai_resolutions = {(512, 512), (768, 768), (1024, 1024), (1280, 720), (1920, 1080)}
        if (w, h) in ai_resolutions:
            score += 25
        # FFT smoothness — AI frames have unnaturally smooth frequency spectra
        gray = np.array(img.convert("L")).astype(np.float32)
        fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
        if np.std(fft_mag) < 3.5:
            score += 30
    except Exception:
        pass
    return min(55, score)


def analyze_video(file_path: str, low_resource: bool = False, deep_scan: bool = False) -> dict:
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
        import time
        t0 = time.time()

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            results['reasons'].append("Could not open video file with OpenCV")
            return results

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        duration = min(duration, 10.0)
        print(f"[VIDEO] ✅ Opened: {duration:.1f}s video | {fps:.0f}fps | {frame_count} frames")

        frames_dir = tempfile.mkdtemp()
        frame_paths = []
        frame_ai_scores = []
        frame_reasons = []
        combined_score = 0

        # Adaptive Sampling: 1 (Ultra-Lite), 8 (Deep/Accurate), 3 (Standard)
        if low_resource:
            num_samples = 1
        elif deep_scan:
            num_samples = 8
        else:
            num_samples = 3
        
        sample_times = np.linspace(0, max(duration - 0.1, 0), num_samples)

        # ── Step 1: Fast heuristic & Laplacian Noise Scan ────────────────────
        print(f"[VIDEO] ⏱ Step 1: Scanning {len(sample_times)} frames...")
        raw_frames = []
        for t in sample_times:
            frame_path = os.path.join(frames_dir, f"frame_{t:.1f}.jpg")
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(frame_path, frame)
                    h_score = _fast_heuristic_score(frame_path)
                    
                    # NEW: Laplacian Noise Check (finds edge discontinuities from morphing)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    # Morphed/compressed frames often have unnaturally low Laplacian variance
                    if laplacian_var < 80: h_score += 20
                    
                    raw_frames.append((t, frame_path, h_score, laplacian_var))
                    print(f"[VIDEO]   Frame {t:.1f}s → heuristic={h_score} (Laplacian={laplacian_var:.1f})")
            except Exception:
                pass

        cap.release()
        print(f"[VIDEO] ✅ Step 1 done: {time.time()-t0:.1f}s elapsed")

        # ── Step 2: Adaptive routing based on frame score variance ───────────
        heuristic_scores = [s for _, _, s, _ in raw_frames]
        score_variance = float(np.std(heuristic_scores)) if len(heuristic_scores) > 1 else 0
        print(f"[VIDEO] ⏱ Step 2: Variance={score_variance:.1f} → {'FULL ViT ALL frames' if score_variance >= 15 else 'ViT on top-2 only'}")

        if score_variance >= 15:
            deep_set = set(fp for _, fp, _, _ in raw_frames)
            frame_reasons.append(
                f"⚠️ High variance (std={score_variance:.1f}) — possible splice. "
                f"Running full ViT on all {len(raw_frames)} frames."
            )
        else:
            sorted_frames = sorted(raw_frames, key=lambda x: x[2], reverse=True)
            deep_set = set(fp for _, fp, _, _ in sorted_frames[:2])
            frame_reasons.append(
                f"Uniform content (std={score_variance:.1f}) — "
                f"deep ViT on 2 key frames, heuristic for rest."
            )

        # ── Step 3: Deep ViT on selected, heuristic on rest ──────────────────
        print(f"[VIDEO] ⏱ Step 3: Deep ViT analysis ({len(deep_set)} frames selected)...")
        for t, frame_path, h_score, lap_res in raw_frames:
            if frame_path in deep_set:
                print(f"[VIDEO]   ViT → Frame {t:.1f}s...")
                vit_t = time.time()
                img_res = analyze_image(frame_path)
                score = img_res['score']
                print(f"[VIDEO]   ViT done: {time.time()-vit_t:.1f}s → score={score}")
                frame_reasons.extend([f"Frame {t:.1f}s: {r}" for r in img_res['reasons']])
            else:
                score = h_score
                print(f"[VIDEO]   Heuristic → Frame {t:.1f}s → score={score}")
            frame_paths.append(frame_path)
            frame_ai_scores.append(score)

        print(f"[VIDEO] ✅ Step 3 done: {time.time()-t0:.1f}s elapsed")

        results['frames_analyzed'] = len(frame_paths)
        results['ai_frame_scores'] = frame_ai_scores

        avg_frame_score = int(np.mean(frame_ai_scores)) if frame_ai_scores else 0

        # SSIM temporal consistency
        ssim_res = {'ssim_mean': 1.0, 'ssim_std': 0.0, 'anomaly': False}
        if not low_resource:
            print(f"[VIDEO] ⏱ Step 4: SSIM computation...")
            ssim_res = _compute_ssim_sequence(frame_paths)
            results['ssim'] = ssim_res
            print(f"[VIDEO] ✅ SSIM done: {time.time()-t0:.1f}s elapsed | anomaly={ssim_res['anomaly']}")
            if ssim_res['anomaly']:
                combined_score += 20
                frame_reasons.append(
                    f"Temporal inconsistency detected (SSIM std={ssim_res['ssim_std']:.3f}) "
                    f"— abrupt structural changes between frames"
                )

            # NEW: Optical Flow Anomaly check
            print(f"[VIDEO] ⏱ Step 5: Optical Flow analysis...")
            flow_res = _compute_optical_flow_anomaly(frame_paths)
            results['optical_flow'] = flow_res
            if flow_res['anomaly']:
                combined_score += 25
                frame_reasons.append(
                    f"Optical Flow Anomaly detected (std={flow_res['flow_std']:.3f}) "
                    f"— unnatural motion warping / pixel sliding identified"
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
            subprocess.run(
                ['ffmpeg', '-i', file_path, '-vn', '-t', '10',
                 '-acodec', 'pcm_s16le', '-ar', '8000', '-ac', '1', audio_path, '-y'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            
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

        # --- Signal Separation (Final Tier Upgrade) ---
        ai_synthesis_score = min(100, core_visual_score)
        # Manipulation score derived from temporal inconsistency + metadata anomalies
        temporal_score = 40 if ssim_res.get('anomaly', False) else 0
        manip_score = min(100, temporal_score + meta_score)

        results['score'] = min(100, int(max(ai_synthesis_score, manip_score, audio_score)))
        results['ai_gen_score'] = ai_synthesis_score
        results['manip_score'] = manip_score
        results['risk_level'] = 'High' if results['score'] >= 60 else 'Medium' if results['score'] >= 30 else 'Low'
        results['reasons'] = frame_reasons[:5] + audio_reasons + meta_res.get('reasons', [])
        return results

    except Exception as e:
        results['reasons'].append(f"Video processing error: {str(e)}")
        return results
