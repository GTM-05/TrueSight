import cv2
import numpy as np
import os
from PIL import Image

def error_level_analysis(image_path, quality=90):
    """
    Performs Error Level Analysis (ELA) to highlight altered segments.
    Resaves the image at a known quality and extracts the absolute difference.
    Heavily manipulated segments (e.g. splicing) will compress differently than the background.
    """
    original = None
    try:
        original = Image.open(image_path).convert('RGB')
        
        # Save temporary compressed version
        temp_filename = "temp_ela.jpg"
        original.save(temp_filename, 'JPEG', quality=quality)
        
        compressed = Image.open(temp_filename)
        
        # Calculate pixel difference
        diff = np.abs(np.array(original).astype(np.int16) - np.array(compressed).astype(np.int16))
        
        # Enhance visual difference
        max_diff = np.max(diff)
        if max_diff == 0:  
            max_diff = 1
            
        scale = 255.0 / max_diff
        enhanced_diff = (diff * scale).astype(np.uint8)
        
        # Save ELA visual map for display in UI
        ela_map_path = "ela_result.jpg"
        Image.fromarray(enhanced_diff).save(ela_map_path)
        
        os.remove(temp_filename)
        
        std_diff = np.std(diff)
        avg_diff = np.mean(diff)
        
        return {
            'std_diff': float(std_diff),
            'avg_diff': float(avg_diff),
            'ela_map_path': ela_map_path
        }
            
    except Exception as e:
        return {'error': str(e)}
    finally:
        if original:
            original.close()

def blur_detection(image_path):
    """
    Calculates the variance of the Laplacian to detect blur.
    Extremely low variance means the image is highly blurred.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {'variance': 0, 'is_blurry': False}
    
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    is_blurry = variance < 100.0
    return {
        'variance': float(variance),
        'is_blurry': is_blurry
    }
    
def analyze_image(image_path):
    """
    Combines ELA and Blur Detection for an offline, heuristic forensic image score.
    """
    score = 0
    reasons = []
    
    ela_res = error_level_analysis(image_path)
    blur_res = blur_detection(image_path)
    
    result = {
        'score': 0,
        'reasons': reasons,
        'metrics': {
            'ela_std_dev': 0,
            'laplacian_variance': blur_res.get('variance', 0),
        },
        'ela_map': ela_res.get('ela_map_path', None),
        'risk_level': 'Low'
    }
    
    if 'error' in ela_res:
        reasons.append(f"ELA Error: {ela_res['error']}")
    else:
        result['metrics']['ela_std_dev'] = ela_res['std_diff']
        # High std dev means non-uniform compression (splicing/forgery)
        if ela_res['std_diff'] > 15.0:
            score += 40
            reasons.append(f"High Error Level Analysis variance (Std Dev: {ela_res['std_diff']:.2f}) - suggests possible image splicing or targeted editing")
        elif ela_res['std_diff'] > 8.0:
            score += 20
            reasons.append(f"Moderate ELA variance (Std Dev: {ela_res['std_diff']:.2f}) - potential minor editing")
            
    if blur_res['is_blurry']:
        score += 20
        reasons.append(f"Image is highly blurry (Laplacian variance {blur_res['variance']:.2f}). This is sometimes done to hide forensic artifacts.")
        
    final_score = min(100, score)
    result['score'] = final_score
    result['risk_level'] = 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'
    
    return result
