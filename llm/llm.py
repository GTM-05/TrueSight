import ollama
import json

def llm_generate_explanation(evidence: dict, final_score: int, verdict: str, confidence: int, model: str = 'qwen2:0.5b', stream: bool = False):
    """
    Expert Mode (Max Speed). Structured data justification.
    """
    img_m = evidence.get('Image', {}).get('metrics', {})
    aud_m = evidence.get('Audio', {}).get('metrics', {})
    vid_d = evidence.get('Video', {})
    
    # Authoritative Forensic Prompt (Optimized for small/quantized models)
    prompt = f"""[FORENSIC_EXPERT_REPORT]
You are a Lead Forensic Analyst. Provide a detailed technical justification for the following results.

DATA:
- Image Score: {evidence.get('Image',{}).get('score',0)}% (ELA Dev: {img_m.get('ela_std_dev',0)})
- Audio Score: {evidence.get('Audio',{}).get('score',0)}% (Pitch Std: {aud_m.get('pitch_std',0)})
- Video Score: {vid_d.get('score',0)}% (Frames: {vid_d.get('frames_analyzed',0)})
- FINAL RISK: {final_score}% ({verdict})
- CONFIDENCE: {confidence}%

ANALYSIS INSTRUCTIONS:
1. EXECUTIVE SUMMARY: Start with a 2-sentence summary of the primary threat vector.
2. TECHNICAL EVIDENCE: Detail specific findings (SSIM anomalies, ELA artifacts, facial synthesis, or frequency inconsistencies).
3. FUSION LOGIC: Explain how the multiple sensors combined to reach the {verdict} risk level.
4. FINAL VERDICT: Conclude with a clinical statement on whether the media is likely AI-generated or manipulated.

Be concise, professional, and use forensic terminology. If a score is low, explain that it represents a lack of detectable artifacts in that modality.
"""

    try:
        if stream:
            return ollama.generate(model=model, prompt=prompt, stream=True, options={'temperature': 0.1, 'num_predict': 250})
        
        response = ollama.generate(
            model=model, prompt=prompt, 
            options={'temperature': 0.1, 'num_predict': 250}
        )
        return response.get('response', "Analysis verified.").strip()
    except Exception:
        return "Manual verification required."
