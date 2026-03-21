import os
import tempfile
from moviepy.editor import VideoFileClip
from modules.image import analyze_image
from modules.audio import analyze_audio
import ollama

def analyze_video(file_path):
    """
    Analyzes a video up to 10 seconds.
    Extracts frames and audio, scores them via existing heuristics, 
    and synthesizes a final score and explanation using Phi-2.
    """
    results = {
        'score': 0,
        'reasons': [],
        'frames_analyzed': 0,
        'audio_analyzed': False,
        'phi_analysis': ''
    }
    
    try:
        clip = VideoFileClip(file_path)
        
        # Limit to 10 seconds
        if clip.duration > 10:
            clip = clip.subclip(0, 10)
            
        video_duration = clip.duration
        
        # Extract 1 frame per second
        frames_dir = tempfile.mkdtemp()
        frame_scores = []
        frame_reasons = set()
        
        for t in range(int(video_duration)):
            frame_path = os.path.join(frames_dir, f"frame_{t}.jpg")
            clip.save_frame(frame_path, t=t)
            
            # Use existing image analysis
            img_res = analyze_image(frame_path)
            frame_scores.append(img_res['score'])
            for r in img_res['reasons']:
                if 'EXIF' not in r and 'Metadata' not in r:
                    frame_reasons.add(f"Frame {t}s: {r}")
                
            os.remove(frame_path)
            
        # Extract audio
        audio_score = 0
        audio_reasons = []
        if clip.audio is not None:
            audio_path = os.path.join(frames_dir, "temp_audio.wav")
            clip.audio.write_audiofile(audio_path, logger=None)
            aud_res = analyze_audio(audio_path)
            audio_score = aud_res['score']
            audio_reasons = aud_res['reasons']
            os.remove(audio_path)
            results['audio_analyzed'] = True
        else:
            # Sora and many AI generators produce completely silent videos
            audio_score = 75
            audio_reasons.append("Missing audio track (highly indicative of raw AI-generated video)")
            results['audio_analyzed'] = True
            
        clip.close()
        try: os.rmdir(frames_dir)
        except: pass
        
        avg_frame_score = sum(frame_scores) / len(frame_scores) if frame_scores else 0
        
        # Compute heuristic combined score
        if results['audio_analyzed']:
            combined_base_score = int((avg_frame_score + audio_score) / 2)
            # Prevent dilution if a stark manipulation flag (like missing audio) was triggered
            if audio_score >= 70:
                combined_base_score = max(combined_base_score, audio_score)
        else:
            combined_base_score = int(avg_frame_score)
            
        # Prepare prompt for Phi-2 synthesis
        reasons_list = list(frame_reasons)[:3] # Keep it concise for prompt
        phi_prompt = f"""
        You are an AI Cyber Forensics Engine. Read the following metric data extracted from a {video_duration}s video clip:
        
        - Avg Visual Anomaly Score: {avg_frame_score:.1f}%
        - Visual Anomalies: {reasons_list if reasons_list else 'None detected'}
        - Audio Anomaly Score: {audio_score}%
        - Audio Anomalies: {audio_reasons if audio_reasons else 'None detected'}
        - Combined Heuristic Risk Score: {combined_base_score}%
        
        Provide a 2-3 sentence professional forensic analysis interpreting these numbers. Conclude with a clear verdict (e.g. Likely Genuine, Suspicious, Likely Deepfake). Be direct and factual based ONLY on metrics provided.
        """
        
        try:
            phi_response = ollama.generate(model='phi', prompt=phi_prompt, options={'temperature': 0.1})
            phi_explanation = phi_response['response'] if phi_response and 'response' in phi_response else "Phi-2 analysis unavailable."
        except Exception as llm_err:
            phi_explanation = f"Phi-2 AI reasoning failed: {str(llm_err)}"
            
        results['score'] = combined_base_score
        results['reasons'] = reasons_list + audio_reasons
        results['frames_analyzed'] = len(frame_scores)
        results['phi_analysis'] = phi_explanation
        
        return results
        
    except Exception as e:
        results['reasons'].append(f"Video processing error: {str(e)}")
        return results
