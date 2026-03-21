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

    threat_found = any(mod.get('threats', {}).get('score', 0) > 0 for mod in all_evidence.values())
    raw_score = (0.35*float(img_score) + 0.25*float(aud_score) + 0.25*float(vid_score) + 0.15*float(url_score))
    final_score = int(min(100.0, max(0.0, raw_score)))
    if threat_found: final_score = 100
    confidence = int(abs((final_score / 100.0) - 0.5) * 2 * 100)
    verdict_level = 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'

    if skip_llm:
        # Generate a structured technical summary for Turbo Mode
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
        'ai_generated_score': int(max(img_ai, aud_score, vid_ai)),
        # FINAL TIER ACCURACY: Boost manipulation score if temporal anomalies are found in video
        'manipulation_score': int(max(img_manip, vid_manip * 1.2, url_score)),
        'final_score': final_score,
        'risk_level': verdict_level,
        'confidence': f"{confidence}%",
        'key_findings': key_findings[:5] if len(key_findings) > 5 else key_findings,
        'ai_explanation': ai_explanation
    }
