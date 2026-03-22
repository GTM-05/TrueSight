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
from PIL import Image

log = logging.getLogger("truesight.video")

class ForensicConfig:
    """Centralized thresholds for the Strong Accurate Algorithm."""
    # Raise thresholds for video frames to avoid compression false positives
    AI_SYNTH_STRONG_THRESHOLD_VIDEO = 65
    AI_SYNTH_EVIDENCE_THRESHOLD_VIDEO = 55
    VIT_MIN_FRAME_AGREEMENT = 0.40

    SAFETY_CAP_SCORE_LIMIT = 75
    SAFETY_CAP_RESULT = 19
    LIVENESS_CONFIRMED_FACTOR = 0.1
    LIVENESS_PARTIAL_FACTOR = 0.7
    MIN_LIVENESS_SIGNALS = 15
    MIN_LIVENESS_FACES = 3
    BLINK_MIN_HUMAN = 2
    LACK_OF_DATA_PENALTY = 15
    LAPLACIAN_BLUR_THRESHOLD = 80

def aggregate_frame_scores(frame_results: list[dict], source: str = "video") -> dict:
    """
    Aggregate per-frame detector results into a single reliable score.
    Eliminates single-frame outliers and deduplicates warnings.
    """
    if not frame_results:
        return {"score": 0, "confidence": 0.1, "is_strong": False, "reasons": []}

    scores = [f["score"] for f in frame_results]
    confs = [f.get("confidence", 0.5) for f in frame_results]

    # Use 75th percentile score to eliminate outliers
    p75_score = float(np.percentile(scores, 75))
    p50_score = float(np.median(scores))
    mean_conf = float(np.mean(confs))

    # Consistency check: high variance = bad frames, not systematic forgery
    score_std = float(np.std(scores))
    consistency = max(0.0, 1.0 - (score_std / (p75_score + 1e-9)))

    # Weight toward p75 if consistent, toward median if variable
    final_score = p75_score * consistency + p50_score * (1.0 - consistency)

    # Deduplicate reasons
    seen = set()
    unique_reasons = []
    for f in frame_results:
        for r in f.get("reasons", []):
            key = r[:60] # Use first 60 chars as dedup key
            if key not in seen:
                seen.add(key)
                unique_reasons.append(r)

    unique_reasons.insert(0, 
        f"[AGGREGATE] {len(frame_results)} frames analyzed — "
        f"p75={p75_score:.1f}, median={p50_score:.1f}, consistency={consistency:.2f}"
    )

    is_strong = p75_score >= ForensicConfig.AI_SYNTH_STRONG_THRESHOLD_VIDEO and consistency > 0.5

    return {
        "score": round(final_score, 1),
        "confidence": round(mean_conf * consistency, 3),
        "is_strong": is_strong,
        "reasons": unique_reasons
    }

# MoviePy is replaced by OpenCV for better performance on 8GB RAM
MOVIEPY_OK = False 

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

from modules.metadata import check_video_metadata
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
    return {
        'ssim_mean': float(round(float(mean), 4)), 
        'ssim_std': float(round(float(std), 4)), 
        'anomaly': anomaly
    }


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
        'flow_std': float(round(float(std), 4)), 
        'anomaly': anomaly, 
        'method': 'Face-Flow' if face_region is not None else 'Global-Flow'
    }


def _get_face_crop(image_path: str) -> dict:
    """
    Detects face and returns a dict with paths to Face, Eye, and Mouth ROIs.
    """
    res = {'face': image_path, 'eyes': None, 'mouth': None, 'is_face': False}
    if not CV2_OK: return res
    
    try:
        from modules.image import _get_face_detector
        face_cascade = _get_face_detector()
        img = cv2.imread(image_path)
        if img is None or face_cascade is None: return res
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # More sensitive detection for video frames (scale 1.05, minNeighbors 3)
        faces = face_cascade.detectMultiScale(gray, 1.05, 3)
        
        # Fallback to Profile Face if frontal fails
        if len(faces) == 0:
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            faces = profile_cascade.detectMultiScale(gray, 1.05, 3)
        
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            # ROIs: Eyes (Top ~1/3), Mouth (Bottom ~1/3)
            eye_roi = img[y+int(h*0.15):y+int(h*0.45), x+int(w*0.1):x+int(w*0.9)]
            mouth_roi = img[y+int(h*0.65):y+int(h*0.95), x+int(w*0.2):x+int(w*0.8)]
            
            face_path = image_path.replace(".jpg", "_f.jpg")
            eye_path = image_path.replace(".jpg", "_e.jpg")
            mouth_path = image_path.replace(".jpg", "_m.jpg")
            
            cv2.imwrite(face_path, img[max(0,y-int(h*0.1)):min(img.shape[0],y+int(h*1.1)), 
                                      max(0,x-int(w*0.1)):min(img.shape[1],x+int(w*1.1))])
            cv2.imwrite(eye_path, eye_roi)
            cv2.imwrite(mouth_path, mouth_roi)
            
            return {'face': face_path, 'eyes': eye_path, 'mouth': mouth_path, 'is_face': True}
    except Exception: pass
    return res


def _analyze_eye_variance(eye_paths: list) -> dict:
    """Detects if eyes are unnaturally static or lack blinks."""
    if not eye_paths or len(eye_paths) < 2: 
        return {'var': 100.0, 'blinks': 0}
    vals = []
    for ep in eye_paths:
        if ep and os.path.exists(ep):
            img = cv2.imread(ep, cv2.IMREAD_GRAYSCALE)
            if img is not None: vals.append(float(np.mean(img)))
    
    if not vals: return {'var': 100.0, 'blinks': 0}
    
    var = float(np.std(vals))
    # Detect blinks: Sharp drops in mean intensity (eyes closing)
    blinks = 0
    if len(vals) > 5:
        # Vectorized derivative check for sharp intensity dips (blinks)
        arr = np.array(vals)
        for i in range(1, len(arr)-1):
            # Sharp dip: current is lower than neighbors by > 7%
            if arr[i] < arr[i-1] * 0.93 and arr[i] < arr[i+1] * 0.93:
                blinks = blinks + 1
    
    return {'var': var, 'blinks': int(blinks)}

def _analyze_mouth_aspect_ratio(mouth_path: str) -> float:
    """Measures mouth aspect ratio (h/w) and edge sharpness."""
    if not mouth_path or not os.path.exists(mouth_path): return 0.0
    img = cv2.imread(mouth_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0.0
    
    # Simple MAR estimation: vertical vs horizontal energy
    h, w = img.shape
    mar = h / w if w > 0 else 0.0
    return float(mar)

def _analyze_mouth_sharpness(mouth_path: str) -> float:
    """Measures mouth edge sharpness (AI teeth are often blurry)."""
    if not mouth_path or not os.path.exists(mouth_path): return 100.0
    img = cv2.imread(mouth_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 100.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _fast_heuristic_score(image_path: str) -> int:
    """Ultra-fast frame pre-scan using np/FFT."""
    score = 0
    try:
        img = Image.open(image_path)
        w, h = img.size
        # Known AI output resolutions
        if (w, h) in {(512, 512), (768, 768), (1024, 1024)}: score += 25
        # FFT smoothness
        gray = np.array(img.convert("L")).astype(np.float32)
        fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-9)
        if np.std(fft_mag) < 2.5: score += 30
    except Exception: pass
    return min(55, score)

def analyze_video(file_path: str, low_resource: bool = False, deep_scan: bool = False) -> dict:
    """
    AI-enhanced video analysis with ROI-based forensics and adaptive sampling.
    """
    if not CV2_OK:
        return {'score': 0, 'risk_level': 'Low', 'reasons': ["OpenCV not available"], 'metrics': {}}

    try:
        import time
        t0 = time.time()

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return {'score': 0, 'risk_level': 'Low', 'reasons': ["File open error"], 'metrics': {}}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = min(frame_count / fps, 10.0)
        
        frames_dir = tempfile.mkdtemp()
        frame_results = []
        eye_paths, skin_rgb, mar_values = [], [], []
        structural_hits = 0
        audio_analyzed = False

        # Increased sampling to 60 for rPPG Nyquist in standard mode (Need ~6fps for heart rate)
        num_samples = 60 if not low_resource else 1
        sample_times = np.linspace(0, max(duration - 0.1, 0), num_samples)

        # ── Step 1: Scanning frames ────────────────────
        for t in sample_times:
            frame_path = os.path.join(frames_dir, f"f_{t:.1f}.jpg")
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(frame_path, frame)
                
                # Biometrics (fast)
                roi = _get_face_crop(frame_path)
                if roi['is_face']:
                    # Mouth Sharpness
                    teeth_v = _analyze_mouth_sharpness(roi['mouth'])
                    # Eye Variance
                    eye_paths.append(roi['eyes'])
                    # Mouth Aspect (for Lip-Sync)
                    mar_values.append(_analyze_mouth_aspect_ratio(roi['mouth']))
                    # Pulse Liveness (CHROM)
                    f_img = cv2.imread(roi['face'])
                    if f_img is not None:
                        b_m = float(np.mean(f_img[:,:,0]))
                        g_m = float(np.mean(f_img[:,:,1]))
                        r_m = float(np.mean(f_img[:,:,2]))
                        skin_rgb.append((r_m, g_m, b_m))
                
                # Lightweight Frame Forensics
                img_res = analyze_image(frame_path, is_video_frame=True)
                frame_results.append(img_res)
                if img_res['metrics'].get('structural_artifact'):
                    structural_hits += 1

        cap.release()

        # ── Step 2: Aggregate Frame Results ────────────
        agg_res = aggregate_frame_scores(frame_results)
        ai_synth = agg_res['score']
        frame_reasons = agg_res['reasons']

        # ── Step 3: SSIM & Flow ─────────────────────────
        frame_paths = [os.path.join(frames_dir, f"f_{t:.1f}.jpg") for t in sample_times] 
        frame_paths = [fp for fp in frame_paths if os.path.exists(fp)]
        
        ssim_res = _compute_ssim_sequence(frame_paths) if not low_resource else {'anomaly': False}
        flow_res = _compute_optical_flow_anomaly(frame_paths) if not low_resource else {'anomaly': False}
        eye_data = _analyze_eye_variance(eye_paths)
        eye_var = eye_data['var']
        blink_count = eye_data['blinks']
        
        # ── Step 4: rPPG Vitality (CHROM Method) ────────
        liveness_anomaly = False
        enough_liveness_data = len(skin_rgb) >= ForensicConfig.MIN_LIVENESS_SIGNALS
        
        if enough_liveness_data:
            rgb = np.array(skin_rgb)
            xs = 3 * rgb[:,0] - 2 * rgb[:,1]
            ys = 1.5 * rgb[:,0] + rgb[:,1] - 1.5 * rgb[:,2]
            sig = (xs / (np.std(xs) + 1e-9)) - (ys / (np.std(ys) + 1e-9))
            sig = sig - np.mean(sig)
            fft_p = np.abs(np.fft.rfft(sig))
            # Peak to Mean Ratio (SNR)
            # Relaxed for compressed video (2.2 instead of 2.8)
            if np.max(fft_p) < np.mean(fft_p) * 2.2:
                liveness_anomaly = True
                
        # ── Step 5: Gaze & Iris Jitter ──────────────────
        iris_jitter_anomaly = False
        if len(eye_paths) >= 5:
            # Human eyes have microsaccades; AI eyes are often frozen or smooth.
            if 0 < eye_var < 0.05:
                iris_jitter_anomaly = True
                frame_reasons.append("Gaze naturalness anomaly (Frozen/Static Iris)")
        
        # ── Step 6: Audio & Lip-Sync Correlation ────────
        audio_score, audio_reasons = 0, []
        audio_path = os.path.join(frames_dir, "aud.wav")
        audio_env = []
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-i', file_path, '-vn', '-t', '10', '-ar', '8000', audio_path, '-y'], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(audio_path):
                aud_res = analyze_audio(audio_path)
                audio_score, audio_reasons = aud_res['score'], aud_res.get('reasons', [])
                audio_analyzed = True
                from modules.audio import get_audio_envelope
                audio_env = get_audio_envelope(audio_path, num_points=len(mar_values))
        except Exception: pass
        
        # ── Step 6.1: AV-Correlation (Lip-Sync Detection) ──
        lip_sync_risk = 0
        if len(mar_values) >= 15 and len(audio_env) == len(mar_values):
            norm_mar = (np.array(mar_values) - np.mean(mar_values)) / (np.std(mar_values) + 1e-9)
            norm_aud = (np.array(audio_env) - np.mean(audio_env)) / (np.std(audio_env) + 1e-9)
            correlation = float(np.mean(norm_mar * norm_aud))
            if correlation < 0.08 and not low_resource:
                lip_sync_risk += 20  # Reduced from 35
                frame_reasons.append(f"AV Weakness: Poor lip-sync correlation ({correlation:.2f})")
            elif correlation < 0.15:
                lip_sync_risk += 10  # Reduced from 20
        
        # ── Step 7: Metadata ───────────────────────────
        meta_res = check_video_metadata(file_path)
        meta_score = meta_res.get('score', 0)
        frame_reasons.extend(meta_res.get('reasons', []))

        # ── Step 8: Scoring Assembly ───────────────────
        manip_score = 0
        if ssim_res.get('anomaly'): manip_score += 25  # Reduced
        if flow_res.get('anomaly'): manip_score += 20  # Reduced
        
        liveness_risk = 0
        if liveness_anomaly: 
            liveness_risk += 20 # Reduced from 35
            frame_reasons.append("Biological Liveness: Human pulse signature was not detected")
        if blink_count == 0 and not low_resource and len(eye_paths) >= 5:
            liveness_risk += 25 # Reduced from 30
            frame_reasons.append("Liveness Anomaly: Zero eye-blinks detected")
        elif eye_var < 0.6:
            liveness_risk += 15 # Reduced from 20

        # ── Step 9: Biological Override ───────────────
        bio_override = False
        has_grids = structural_hits >= (num_samples * 0.2) # 20% of frames
        
        # More lenient override: Accept 1 blink or partial pulse if technical signals are low
        is_liveness_partial = not liveness_anomaly or blink_count >= 1
        
        if is_liveness_partial and ai_synth < ForensicConfig.AI_SYNTH_STRONG_THRESHOLD_VIDEO and not has_grids:
            bio_override = True
            reduction = ForensicConfig.LIVENESS_CONFIRMED_FACTOR if (not liveness_anomaly and blink_count >= 1) else ForensicConfig.LIVENESS_PARTIAL_FACTOR
            ai_synth = int(ai_synth * reduction)
            audio_score = int(audio_score * reduction)
            meta_score = int(meta_score * reduction)
            manip_score = int(manip_score * reduction)
            lip_sync_risk = int(lip_sync_risk * reduction)

        if bio_override:
            frame_reasons.append("Verification: Multimodal liveness confirms biological authenticity.")
        
        if not enough_liveness_data:
            liveness_risk = ForensicConfig.LACK_OF_DATA_PENALTY
            
        final_score = int(min(100, max(ai_synth, manip_score + liveness_risk + lip_sync_risk, audio_score, meta_score)))
        
        # TIER 3: Low-Confidence Filtering (Safety Floor)
        has_strong_evidence = ai_synth > ForensicConfig.AI_SYNTH_EVIDENCE_THRESHOLD_VIDEO or has_grids or (enough_liveness_data and liveness_anomaly)
        
        if final_score < ForensicConfig.SAFETY_CAP_SCORE_LIMIT and not has_strong_evidence:
            final_score = min(final_score, ForensicConfig.SAFETY_CAP_RESULT)
            
        # Final Verdict
        res = {
            'score': int(final_score),
            'ai_gen_score': int(ai_synth),
            'manip_score': int(max(manip_score + liveness_risk + lip_sync_risk, audio_score)),
            'risk_level': 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low',
            'reasons': sorted(list(set(frame_reasons + list(audio_reasons)))), 
            'metadata': meta_res,
            'metrics': {
                'ai_probability': int(ai_synth),
                'temporal_anomaly': bool(ssim_res['anomaly']),
                'audio_score': int(audio_score),
                'meta_score': int(meta_score),
                'structural_artifact': bool(has_grids),
                'blink_count': int(blink_count),
                'liveness_override': bool(bio_override),
                'iris_jitter_anomaly': bool(iris_jitter_anomaly),
                'liveness_anomaly': bool(liveness_anomaly),
                'enough_liveness_data': bool(enough_liveness_data)
            }
        }

        # Cleanup
        for fp in frame_paths:
            if os.path.exists(fp): 
                try: os.remove(fp)
                except: pass
        if os.path.exists(audio_path):
            try: os.remove(audio_path)
            except: pass
        try: os.rmdir(frames_dir)
        except: pass

        return res

    except Exception as e:
        log.error(f"Error: {str(e)}")
        return {'score': 0, 'risk_level': 'Low', 'reasons': [f"Critical error: {str(e)}"], 'metrics': {}}
