from llm.phi3 import llm_generate_explanation

def generate_final_verdict_ai(all_evidence: dict) -> dict:
    """
    Fusion Engine (Weighted Mathematical Logic).
    DOES NOT use LLM for decision making. Final score is computed explicitly.
    LLM is only used to format the final narrative.
    """
    if not all_evidence:
        return {
            'threat_score': 0,
            'ai_generated_score': 0,
            'manipulation_score': 0,
            'final_score': 0,
            'risk_level': 'Low',
            'confidence': 'Low',
            'key_findings': [],
            'verdict': 'Low',
            'ai_explanation': 'No data analyzed.'
        }

    # Extract individual scores
    img_score = all_evidence.get('Image Analysis', {}).get('score', 0)
    aud_score = all_evidence.get('Audio Analysis', {}).get('score', 0)
    vid_score = all_evidence.get('Video Analysis', {}).get('score', 0)
    url_score = all_evidence.get('URL Analysis', {}).get('score', 0)
    threat_found = all_evidence.get('Video Analysis', {}).get('threats', {}).get('score', 0) > 0 or \
                   all_evidence.get('Image Analysis', {}).get('threats', {}).get('score', 0) > 0

    # Normalization & Weights (as explicitly requested)
    # If a module wasn't run, we shouldn't penalize it to 0, but for this strict mathematical model:
    # If the module wasn't run, we just use 0, but ideally we'd re-weight. Let's strictly follow the formula:
    
    # Calculate weighted final score
    final_score = (0.35 * img_score) + (0.25 * aud_score) + (0.25 * vid_score) + (0.15 * url_score)
    final_score = int(min(100, max(0, final_score)))
    
    # Override if threat found
    if threat_found:
        final_score = max(final_score, 100)
    
    # Compute Confidence %
    confidence = int(abs((final_score / 100.0) - 0.5) * 2 * 100)
    
    # Determine Verdict
    verdict = 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'
    
    # Sub-scores for compatibility with app.py UI expectations
    max_manipulation = max(img_score, vid_score)
    max_ai = max(img_score, aud_score, vid_score)
    
    # Generate Explanation via Phi-3
    ai_explanation = llm_generate_explanation(all_evidence, final_score, verdict, confidence)

    # Compile key findings heuristically for the UI
    key_findings = []
    for mod, data in all_evidence.items():
        reasons = data.get('reasons', [])
        for r in reasons:
            if r and "low" not in r.lower():
                key_findings.append(f"[{mod}] {r}")
    
    return {
        'threat_score': 100 if threat_found else 0,
        'ai_generated_score': int(max_ai),
        'manipulation_score': int(max_manipulation),
        'final_score': final_score,
        'risk_level': verdict,
        'confidence': f"{confidence}%",
        'key_findings': key_findings[:5],
        'verdict': verdict,
        'ai_explanation': ai_explanation,
        'summary': f"Mathematical Fusion: Image({0.35}), Audio({0.25}), Video({0.25}), URL({0.15}). Overrides active: {threat_found}."
    }
