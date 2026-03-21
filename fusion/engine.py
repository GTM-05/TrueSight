from llm.phi2 import llm_reason_verdict

def generate_final_verdict(all_evidence: dict) -> dict:
    """
    Fuses the individual module results into a structured final verdict.
    Uses the LLM to reason over all evidence rather than a simple numeric average.
    Falls back to averaging if Ollama is unavailable.
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
            'summary': 'No data analyzed.'
        }

    verdict = llm_reason_verdict(all_evidence)

    # Normalize: map 'verdict' string to 'risk_level' for backward compatibility
    risk_level = verdict.get('verdict', 'Low')
    final_score = verdict.get('final_score', 0)

    verdict['risk_level'] = risk_level
    verdict['summary'] = (
        f"Analyzed {len(all_evidence)} modalities. "
        f"Threat: {verdict.get('threat_score', 0)}% | "
        f"AI-Generated: {verdict.get('ai_generated_score', 0)}% | "
        f"Manipulation: {verdict.get('manipulation_score', 0)}% | "
        f"Overall: {risk_level} ({final_score}%)"
    )

    return verdict
