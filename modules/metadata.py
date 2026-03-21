import os
import exifread

SUSPICIOUS_SOFTWARE = ['photoshop', 'gimp', 'lightroom', 'canva', 'fotor', 'paint', 'snapseed', 'illustrator', 'coreldraw']

def check_metadata(image_path):
    """
    Extracts EXIF metadata and analyzes for signs of manipulation.
    Returns a forensic risk score based on missing or altered data.
    """
    score = 0
    reasons = []
    metadata = {}
    
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            if not tags:
                from PIL import Image
                try:
                    with Image.open(image_path) as img:
                        w, h = img.size
                        # Common generative AI resolutions
                        if (w, h) in [(1024, 1024), (512, 512), (1024, 1792), (1792, 1024), (1024, 768), (768, 1024)]:
                            score += 75
                            reasons.append(f"CRITICAL: No EXIF data AND exact AI-generation resolution ({w}x{h}). Highly indicative of Midjourney/DALL-E/Stable Diffusion output.")
                            return {'score': score, 'reasons': reasons, 'metadata': {}, 'risk_level': 'High'}
                except:
                    pass
                    
                score += 30
                reasons.append("No EXIF metadata found. Often stripped during manipulation or downloading from social media/messaging apps.")
                return {'score': score, 'reasons': reasons, 'metadata': {}, 'risk_level': 'Medium'}
            for tag in tags.keys():
                if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                    metadata[tag] = str(tags[tag])
                    
            # Check for specific tags indicating editing software
            software_tags = ['Image Software', 'Software']
            for stag in software_tags:
                if stag in metadata:
                    software = metadata[stag].lower()
                    if any(s in software for s in SUSPICIOUS_SOFTWARE):
                        score += 50
                        reasons.append(f"Edited with known image manipulation software: {metadata[stag]}")
                        break
                        
            # Check datetime anomalies (Original vs Digitized vs Modified)
            dt_orig = metadata.get('EXIF DateTimeOriginal')
            dt_mod = metadata.get('Image DateTime')
            
            if dt_orig and dt_mod and dt_orig != dt_mod:
                score += 20
                reasons.append(f"Modification timestamp ({dt_mod}) differs from original capture time ({dt_orig})")
                
            # Missing camera info when claiming to be an original can also be a red flag (optional check)
            if 'Image Model' not in metadata and 'Image Make' not in metadata:
                score += 10
                reasons.append("Missing Camera Make/Model in EXIF, unusual for unedited raw photos.")
                
    except Exception as e:
        score += 10
        reasons.append(f"Error parsing metadata: {str(e)}")
        
    final_score = min(100, score)
    return {
        'score': final_score,
        'reasons': reasons,
        'metadata': metadata,
        'risk_level': 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'
    }

def check_video_metadata(video_path):
    """
    Analyzes video metadata using ffprobe for signs of AI generation or editing.
    """
    score = 0
    reasons = []
    metadata = {}
    tags = {}  # ensure always defined even if try block fails early

    try:
        import subprocess
        import json
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        
        fmt = data.get('format', {})
        tags = fmt.get('tags', {})
        
        # 1. Check for suspicious encoders
        encoder = tags.get('encoder', '').lower()
        if any(s in encoder for s in ['lavc', 'ffmpeg', 'handbrake', 'premiere', 'resolve']):
            # Not necessarily AI, but common in re-encoded/manipulated video
            score += 15
            reasons.append(f"Video re-encoded with potentially suspicious software: {encoder}")
            
        # 2. Check for missing metadata common in AI generators
        if not tags or len(tags) < 2:
            score += 35
            reasons.append("Highly suspicious: Sparse or missing global metadata tags. Common in raw AI video output.")
            
        # 3. Check for specific AI generator markers (rare but possible)
        if any(k in str(tags).lower() for k in ['sora', 'runway', 'pika', 'stable-video']):
            score += 90
            reasons.append("CRITICAL: AI Generator signatures found in metadata.")

    except Exception as e:
        score += 10
        reasons.append(f"Metadata scan failed: {str(e)}")
        
    final_score = min(100, score)
    return {
        'score': final_score,
        'reasons': reasons,
        'metadata': tags,
        'risk_level': 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'
    }
