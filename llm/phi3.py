import ollama
import json

def llm_generate_explanation(evidence: dict, final_score: int, verdict: str, confidence: int) -> str:
    """
    Acts exclusively as the Explanation Layer (CPU-Optimized).
    Does not make decisions. Interprets the numeric fusion verdict.
    """
    
    # Safely extract metrics from the evidence payload
    img_data = evidence.get('Image Analysis', {})
    aud_data = evidence.get('Audio Analysis', {})
    vid_data = evidence.get('Video Analysis', {})
    url_data = evidence.get('URL Analysis', {})
    
    img_metrics = img_data.get('metrics', {})
    aud_metrics = aud_data.get('metrics', {})
    vid_ssim = vid_data.get('ssim', {})
    
    prompt = f"""You are a certified cyber forensic analyst.

IMPORTANT RULES:
- You DO NOT decide whether the content is fake or real.
- The verdict is already given — you must only justify it.
- Base your explanation strictly on the numerical evidence provided.
- Do NOT hallucinate or add assumptions.
- Keep the explanation professional, precise, and technical.

----------------------------------------

CASE DATA:

Modality Scores:
- Image Score: {img_data.get('score', 0)}
- Audio Score: {aud_data.get('score', 0)}
- Video Score: {vid_data.get('score', 0)}
- URL Score: {url_data.get('score', 0)}

Key Indicators:
- ELA (Error Level Analysis): {img_metrics.get('ela_std_dev', 0)}
- Noise/Blur Inconsistency: {img_metrics.get('laplacian_variance', 0)}
- AI Image Probability: {img_metrics.get('ai_probability', 0)}

Audio Features:
- Pitch Stability: {aud_metrics.get('pitch_std', 0)}
- Energy Consistency: {aud_metrics.get('rms_std', 0)}
- MFCC Delta: {aud_metrics.get('mfcc_delta_std', 0)}

Video Features:
- Temporal Consistency (SSIM): {vid_ssim.get('ssim_std', 0)}
- Average AI Frame Score: {int(sum(vid_data.get('ai_frame_scores', [0])) / max(len(vid_data.get('ai_frame_scores', [1])), 1)) if 'ai_frame_scores' in vid_data else 0}

Final Computed Score: {final_score}
Final Verdict: {verdict}
Confidence: {confidence}%

----------------------------------------

TASK:

Generate a structured forensic explanation with the following sections:

1. Summary (2–3 lines):
   - Briefly explain why the given verdict is justified.

2. Technical Analysis:
   - Explain key anomalies from image, audio, and video.
   - Mention only relevant indicators (do not list everything blindly).
   - Use clear technical reasoning.

3. Conclusion:
   - Reinforce why the verdict is correct based on evidence.

----------------------------------------

OUTPUT FORMAT (STRICT):

Summary:
<your answer>

Technical Analysis:
<your answer>

Conclusion:
<your answer>
"""
    
    fallback_text = "Summary:\nAnalysis completed heuristically.\n\nTechnical Analysis:\nLLM was bypassed or unavailable. Please review numeric scores directly.\n\nConclusion:\nBased on numeric fusion engine score."
    
    try:
        response = ollama.generate(
            model='phi3:mini', 
            prompt=prompt, 
            options={'temperature': 0.1, 'num_predict': 400}
        )
        return response.get('response', fallback_text).strip()
    except Exception:
        return fallback_text
