from llm.llm import llm_generate_explanation

def generate_final_verdict_ai(all_evidence: dict, skip_llm: bool = False, model: str = 'qwen2:0.5b', stream: bool = False) -> dict:
    """
    Fusion Engine (Weighted Mathematical Logic).
    """
    if not all_evidence:
        return {'final_score': 0, 'risk_level': 'Low', 'ai_explanation': 'No data.'}

    # ... [Same scoring logic] ...
    img_score = all_evidence.get('Image', {}).get('score', 0)
    aud_score = all_evidence.get('Audio', {}).get('score', 0)
    vid_score = all_evidence.get('Video', {}).get('score', 0)
    url_score = all_evidence.get('URL', {}).get('score', 0)
    # Extract sub-scores if available (Final Tier)
    img_ai = all_evidence.get('Image', {}).get('ai_gen_score', img_score)
    img_manip = all_evidence.get('Image', {}).get('manip_score', img_score)
    vid_ai = all_evidence.get('Video', {}).get('ai_gen_score', vid_score)
    vid_manip = all_evidence.get('Video', {}).get('manip_score', vid_score)

    # Optional: Extract flicker and lip-sync scores from Video evidence if present
    vid_flicker = all_evidence.get('Video', {}).get('flicker_score', 0)
    vid_lip = all_evidence.get('Video', {}).get('lip_sync_score', 0)
    vid_meta = all_evidence.get('Video', {}).get('meta_score', 0)
    vid_int_aud = all_evidence.get('Video', {}).get('internal_audio_score', 0)

    threat_found = any(mod.get('threats', {}).get('score', 0) > 0 for mod in all_evidence.values())
    
    # --- Max-Biased Fusion Optimization ---
    # weighted_avg is the traditional baseline
    weighted_avg = (0.35*float(img_score) + 0.25*float(aud_score) + 0.25*float(vid_score) + 0.15*float(url_score))
    
    # max_module_score is the single most suspicious finding
    # We include vid_flicker and vid_lip in the max check
    max_module_score = float(max(img_score, aud_score, vid_score, url_score, vid_flicker, vid_lip))
    
    # If any module is high risk (>=60), the final score should reflect that high risk,
    # rather than being averaged down by unrelated components.
    if max_module_score >= 60:
        # Heavily bias towards the high-risk finding
        raw_score = max_module_score * 0.9 + (weighted_avg * 0.1)
    elif max_module_score >= 30:
        # Medium risk bias
        raw_score = max_module_score * 0.7 + (weighted_avg * 0.3)
    else:
        raw_score = weighted_avg

    final_score = int(min(100.0, max(0.0, raw_score)))
    if threat_found: final_score = 100
    confidence = int(abs((final_score / 100.0) - 0.5) * 2 * 100)
    verdict_level = 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'

    if skip_llm:
        # ... [TURBO summary logic remains same] ...
        summary = f"### [TURBO] FORENSIC TECHNICAL SUMMARY\n\n"
        summary += f"**Final Risk Score:** {final_score}% ({verdict_level} Threat Detected)\n"
        summary += f"**Confidence Level:** {confidence}%\n\n"
        summary += "#### Technical Breakdown:\n"
        for mod, data in all_evidence.items():
            m_score = data.get('score', 0)
            summary += f"- **{mod}**: Analyzed with {m_score}% risk. Indicators: {', '.join(data.get('reasons', ['None']))}\n"
        summary += "\n**Verdict:** Manual review of high-risk vectors is recommended. AI Analyst reasoning was bypassed for speed."
        ai_explanation = summary
    else:
        ai_explanation = llm_generate_explanation(all_evidence, final_score, verdict_level, confidence, model=model, stream=stream)

    key_findings = []
    for mod, data in all_evidence.items():
        reasons = list(data.get('reasons', []))
        for r in reasons:
            if r: key_findings.append(f"[{mod}] {r}")

    return {
        'threat_score': 100 if threat_found else 0,
        'ai_generated_score': int(max(img_ai, aud_score, vid_ai, vid_int_aud)),
        # FINAL TIER ACCURACY: Boost manipulation score if temporal/flicker/lip-sync anomalies are found
        # Include metadata score in manipulation/suspicion category
        'manipulation_score': int(max(img_manip, vid_manip, url_score, vid_flicker * 1.1, vid_lip * 1.1, vid_meta)),
        'final_score': final_score,
        'risk_level': verdict_level,
        'confidence': f"{confidence}%",
        'key_findings': key_findings[:5] if len(key_findings) > 5 else key_findings,
        'ai_explanation': ai_explanation
    }
