import numpy as np
from llm.llm import llm_generate_explanation

def generate_final_verdict_ai(all_evidence: dict, skip_llm: bool = False, model: str = 'qwen2:0.5b', stream: bool = False) -> dict:
    """
    Fusion Engine (Weighted Mathematical Logic) — Phase 3 Refinement.
    Implements Confidence-Weighted Fusion and Cross-Modal Disagreement Detection.
    """
    if not all_evidence:
        return {'final_score': 0, 'risk_level': 'Low', 'ai_explanation': 'No data.'}

    # ── 1. Data Extraction & Module Scoring ──────────────────────────────────
    img_data = all_evidence.get('Image', {})
    aud_data = all_evidence.get('Audio', {})
    vid_data = all_evidence.get('Video', {})
    url_data = all_evidence.get('URL', {})

    img_score = float(img_data.get('score', 0))
    aud_score = float(aud_data.get('score', 0))
    vid_score = float(vid_data.get('score', 0))
    url_score = float(url_data.get('score', 0))

    # Sub-scores for risk breakdown
    img_ai = img_data.get('ai_gen_score', img_score)
    img_manip = img_data.get('manip_score', img_score)
    vid_ai = vid_data.get('ai_gen_score', vid_score)
    vid_manip = vid_data.get('manip_score', vid_score)
    vid_flicker = vid_data.get('flicker_score', 0)
    vid_lip = vid_data.get('lip_sync_score', 0)

    threat_found = any(mod.get('threats', {}).get('score', 0) > 0 for mod in all_evidence.values())
    
    # ── 2. Confidence-Weighted Max Fusion ──────────────────────────────────────
    # We weight each score by its reported confidence. 
    # AI models and biological checks (rPPG) provide higher confidence than ELA/Blur.
    weighted_sum = 0
    total_weight = 0
    meta_reasons = []
    
    module_signals = []

    for mod_name, data in all_evidence.items():
        score = float(data.get('score', 0))
        conf = float(data.get('confidence', 0.5))
        is_strong = data.get('is_strong', False)
        
        # Signal is the confidence-adjusted score
        # Strong indicators get a 'reliability boost'
        signal = score * (conf * (1.5 if is_strong else 1.0))
        module_signals.append(signal)

        # Weight for the average
        weight = conf * (2.0 if is_strong else 1.0)
        weighted_sum += score * weight
        total_weight += weight

    weighted_avg = weighted_sum / (total_weight + 1e-9)
    # The true 'max' should be the highest reliability-adjusted signal
    max_adj_score = max(module_signals) if module_signals else 0.0
    
    # vid_flicker/vid_lip are already heuristic sub-scores, let's treat them as mid-confidence
    secondary_max = max(float(vid_flicker) * 0.7, float(vid_lip) * 0.7)
    max_score = max(max_adj_score, secondary_max)
    
    # Fusion Logic: Start with adjusted Max, but moderated by Weighted Average
    if max_score >= 60:
        base_score = max_score * 0.8 + weighted_avg * 0.2
    elif max_score >= 30:
        base_score = max_score * 0.6 + weighted_avg * 0.4
    else:
        base_score = weighted_avg

    # ── 3. Cross-Modal Corroboration (Disagreement Detection) ────────────────
    disagreement_bonus = 0
    # Red Flag: Video looks real (Liveness OK) but Audio is synthetic/monotone
    if vid_score < 25 and aud_score > 55:
        disagreement_bonus = 30
        meta_reasons.append("Cross-Modal Disagreement: High AI-Audio signature matched with likely-human Video evidence.")
    # Red Flag: Video is AI (Flicker/Spectral) but Audio is clean
    elif vid_score > 55 and aud_score < 20:
        disagreement_bonus = 15
        meta_reasons.append("Cross-Modal Disagreement: Synthetic Video artifacts with clean human-like Audio.")

    final_score = base_score + disagreement_bonus

    # ── 4. Graduated Safety Floor ─────────────────────────────────────────────
    # Instead of a hard 19% cap, we only suppress "background noise" if there's
    # zero high-confidence evidence and no modal disagreement.
    has_strong = any(data.get('is_strong') for data in all_evidence.values())
    
    if final_score < 19 and not has_strong and disagreement_bonus == 0:
        # Keep low-level heuristics quiet unless they agree or are strong
        final_score = min(final_score, 19)
    
    final_score = int(min(100, max(0, final_score)))

    # ── 5. Elite Overrides ────────────────────────────────────────────────────
    img_met = img_data.get('metrics', {})
    vid_met = vid_data.get('metrics', {})
    
    if abs(img_met.get('spectral_slope', -2.2) + 2.2) > 0.6:
        final_score = max(final_score, 75)
    if img_met.get('chrom_score', 10.0) < 2.2 and img_score > 25:
        final_score = max(final_score, 68)
    if vid_met.get('iris_jitter_anomaly'):
        final_score = max(final_score, 72)
    
    if threat_found: final_score = 100

    # ── 6. Reporting & Explanation ───────────────────────────────────────────
    verdict_level = 'High' if final_score >= 60 else 'Medium' if final_score >= 30 else 'Low'
    # Confidence is the mean of reporting module confidences
    raw_conf = np.mean([float(d.get('confidence', 0.5)) for d in all_evidence.values()])
    display_conf = int(raw_conf * 100)

    if skip_llm:
        summary = f"### [TURBO] FORENSIC FUSION REPORT\n\n"
        summary += f"**Verdict:** {verdict_level} Risk ({final_score}%)\n"
        summary += f"**Confidence:** {display_conf}%\n\n"
        summary += "#### Reasoning:\n"
        for r in meta_reasons: summary += f"- ⚠️ {r}\n"
        for mod, data in all_evidence.items():
            summary += f"- **{mod}**: Score {data.get('score', 0)}% — {', '.join(data.get('reasons', ['Clean']))[:150]}...\n"
        ai_explanation = summary
    else:
        ai_explanation = llm_generate_explanation(all_evidence, final_score, verdict_level, display_conf, model=model, stream=stream)

    key_findings = meta_reasons + [f"[{m}] {r}" for m, d in all_evidence.items() for r in d.get('reasons', []) if r]

    return {
        'threat_score': 100 if threat_found else 0,
        'ai_generated_score': int(max(float(img_ai), float(aud_score), float(vid_ai))),
        'manipulation_score': int(max(float(img_manip), float(vid_manip), float(url_score), float(vid_flicker))),
        'final_score': int(final_score),
        'risk_level': str(verdict_level),
        'confidence': f"{display_conf}%",
        'key_findings': list(key_findings[:5]),
        'ai_explanation': str(ai_explanation)
    }
