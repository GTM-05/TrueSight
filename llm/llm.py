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
You are a Lead Forensic Analyst. Provide a detailed, professional narrative based on these results.
DATA: {json.dumps({
    'img': {'score': evidence.get('Image',{}).get('score',0), 'ela': img_m.get('ela_std_dev',0)},
    'aud': {'score': evidence.get('Audio',{}).get('score',0), 'pitch_std': aud_m.get('pitch_std',0)},
    'vid': {'score': vid_d.get('score',0)},
    'fusion': {'total': final_score, 'level': verdict, 'confidence': f"{confidence}%"}
})}
INSTRUCTIONS:
1. Start with a clear EXECUTIVE SUMMARY (2-3 sentences).
2. Detail the TECHNICAL EVIDENCE (specifically mentioning ELA, Spectral grid, or pitch artifacts).
3. Conclude with a FINAL VERDICT based on mathematical fusion.
Be clinical, certain, and use forensic terminology."""

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
