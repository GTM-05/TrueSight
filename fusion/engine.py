"""
TrueSight — Fusion Engine v3.0 (FIX-1, FIX-2, FIX-6, FIX-8)
Strict execution order: fuse → cross-modal penalty → liveness reduction → safety floor.
"""

from __future__ import annotations

from typing import Any, Optional

from config import CFG
from llm.llm import llm_generate_explanation


def fuse_module_results(detectors: dict[str, Any]) -> float:
    """
    FIX-1: Strong anchors dominate; weak signals add bounded boost only.
    """
    strong = [
        d
        for d in detectors.values()
        if d and d.get("is_strong") and float(d.get("confidence", 0) or 0) > 0.5
    ]

    if strong:
        base = max(float(d["score"]) for d in strong)
        boost = min(
            sum(
                float(d["score"])
                * float(d.get("confidence", 0.5) or 0.5)
                * CFG.FUSION_BOOST_MULTIPLIER
                for d in detectors.values()
                if d and (not d.get("is_strong")) and float(d.get("score", 0) or 0) > 10
            ),
            CFG.FUSION_BOOST_MAX,
        )
        return min(base + boost, 100.0)

    weighted = [
        (
            float(d["score"]) * float(d.get("confidence", 0.5) or 0.5),
            float(d.get("confidence", 0.5) or 0.5),
        )
        for d in detectors.values()
        if d
        and float(d.get("confidence", 0) or 0) > 0.3
        and float(d.get("score", 0) or 0) > 0
    ]
    if not weighted:
        return 10.0
    scores, confs = zip(*weighted)
    return float(sum(scores) / sum(confs))


def cross_modal_penalty(
    video_s: float,
    audio_s: float,
    image_s: float,
    video_c: float,
    audio_c: float,
    image_c: float,
) -> tuple[float, Optional[str]]:
    """FIX-6: Penalize large spread between modalities with sufficient confidence."""
    active: dict[str, float] = {}
    conf_map = {"video": video_c, "audio": audio_c, "image": image_c}
    score_map = {"video": video_s, "audio": audio_s, "image": image_s}
    for k in score_map:
        score = float(score_map[k] or 0)
        conf = float(conf_map[k] or 0)
        # Higher confidence required for low scores; lower confidence okay for clear forensics
        if conf > 0.4 or (score > 60 and conf > 0.3):
            active[k] = score

    if len(active) < 2:
        return 0.0, None

    spread = max(active.values()) - min(active.values())

    if spread > CFG.CROSS_MODAL_SPREAD_STRONG:
        penalty = min(spread * 0.4, CFG.CROSS_MODAL_PENALTY_MAX)
        reason = (
            f"[CROSS-MODAL] Modality disagreement (spread={spread:.0f} pts): {active}. "
            f"Characteristic of single-track deepfake substitution."
        )
        return penalty, reason

    if spread > CFG.CROSS_MODAL_SPREAD_WEAK:
        penalty = spread * 0.2
        return penalty, f"[CROSS-MODAL] Mild inconsistency (spread={spread:.0f} pts)."

    return 0.0, None


def apply_liveness_reduction(
    raw_score: float, liveness: dict[str, Any], all_signals: list[dict]
) -> tuple[float, list[str]]:
    """
    FIX-2: Liveness reduction must be GATED on other signals.
    """
    if not liveness.get("liveness_detected"):
        return raw_score, []

    pulse = bool(liveness.get("pulse_confirmed", False))
    blinks = int(liveness.get("blink_count", 0) or 0)
    jitter = float(liveness.get("iris_jitter", 0.0) or 0.0)
    conf = float(liveness.get("confidence", 0.5) or 0.5)

    def _is_structural(s):
        score = float(s.get("score", 0) or 0)
        if score >= 60: return True # High confidence forensic finding
        if score < 25: return False
        reasons = s.get("reasons", [])
        if isinstance(reasons, str): reasons = [reasons]
        txt = " ".join(reasons).lower()
        return any(t in txt for t in ["ela", "srm", "ai", "morph", "spectrum", "dct", "warp", "consensus"])

    def _is_any_forensic(s):
        score = float(s.get("score", 0) or 0)
        if score < 15: return False
        reasons = s.get("reasons", [])
        if isinstance(reasons, str): reasons = [reasons]
        txt = " ".join(reasons).lower()
        return "liveness" not in txt or score > 40

    structural_fired = sum(1 for s in all_signals if _is_structural(s))
    other_fired = sum(1 for s in all_signals if _is_any_forensic(s))

    # DEBUG
    with open("/tmp/fusion_debug.log", "a") as f:
        f.write(f"DEBUG: structural_fired={structural_fired}, other_fired={other_fired}\n")

    # FIX-2: If structural signals are firing, HEAVILY limit or skip liveness reduction
    if structural_fired >= 1:
        # Deepfakes often hallucinate/preserve pulse; don't let it mask forensics
        factor = 1.0 
        msg = f"Liveness reduction SKIPPED — {structural_fired} structural signals active (possible pulse spoofing)."
    elif other_fired >= 3:
        factor = 0.70
        msg = f"Partial liveness reduction only — {other_fired} forensic signals active. 30% reduction applied."
    elif other_fired >= 1:
        factor = 0.40
        msg = f"Significant liveness reduction — only {other_fired} heuristic signals active. 60% reduction applied."
    elif pulse and blinks >= 2 and jitter > float(CFG.IRIS_JITTER_MIN):
        factor = float(CFG.LIVENESS_CONFIRMED_FACTOR)
        msg = "Full liveness confirmed, no other signals — 90% reduction applied."
    elif pulse or blinks >= 1:
        factor = float(CFG.LIVENESS_PARTIAL_FACTOR)
        msg = f"Partial liveness (pulse={pulse}, blinks={blinks}) — 50% reduction."
    else:
        return raw_score, []

    if factor >= 1.0:
        return raw_score, [f"[LIVENESS] {msg}"]

    reduced = max(raw_score * factor * conf, float(CFG.LIVENESS_MIN_FLOOR))
    return reduced, [f"[LIVENESS] {msg}"]


def compute_morphing_score(video_result: dict[str, Any], audio_result: dict[str, Any]) -> float:
    """
    Unified morphing / manipulation index: face SSIM dual-threshold, face warp,
    ffprobe metadata hints, and audio phase splice spike density.
    All weights and caps come from CFG.
    """
    mc = video_result.get("morph_components") or {}
    ssim_s = float(mc.get("ssim_morph", 0) or 0)
    warp_s = float(mc.get("face_warp", 0) or 0)
    spatial = max(ssim_s, warp_s) * float(CFG.MORPHING_SPATIAL_WEIGHT)
    color_s = float(mc.get("color_score", 0) or 0)
    color_m = min(25.0, color_s * 0.8) # Cap color contribution to morphing index

    meta = float((video_result.get("metrics") or {}).get("meta_score", 0) or 0)
    meta_m = min(
        float(CFG.MORPHING_META_CAP),
        meta * float(CFG.MORPHING_META_SCALE),
    )

    ph = (audio_result.get("sub_scores") or {}).get("phase") or {}
    spikes = int(ph.get("spike_count", 0) or 0)
    phase_m = min(
        float(CFG.MORPHING_PHASE_SCORE_CAP),
        spikes * float(CFG.MORPHING_PHASE_POINTS_PER_SPIKE),
    )

    return float(min(100.0, spatial + color_m + meta_m + phase_m))


def apply_safety_floor(
    raw_score: float, signals: list[dict], reasons: list[str]
) -> float:
    """
    FIX-1: Accumulation-aware graduated safety floor.
    """
    fired_strong = [
        s
        for s in signals
        if s and s.get("is_strong") and float(s.get("confidence", 0) or 0) > 0.4
    ]
    fired_medium = [
        s
        for s in signals
        if s and float(s.get("score", 0) or 0) > 25 and float(s.get("confidence", 0) or 0) > 0.3
    ]
    fired_weak = [
        s for s in signals if s and 10 < float(s.get("score", 0) or 0) <= 25
    ]

    fired_count = len(fired_medium) + len(fired_weak)

    # Hard floor from strongest single signal
    if fired_strong:
        max_strong = max(float(s.get("score", 0) or 0) for s in fired_strong)
        hard_floor = max_strong * 0.65
        raw_score = max(raw_score, hard_floor)

    if raw_score >= CFG.SAFETY_CAP_SCORE_LIMIT:
        return raw_score

    if fired_strong:
        return max(raw_score, CFG.SAFETY_FLOOR_STRONG)

    # FIX-1: Accumulation floor
    if fired_count >= 4:
        reasons.append(
            f"[ACCUM] {fired_count} detectors fired simultaneously — accumulation pattern."
        )
        return max(raw_score, float(CFG.ACCUM_FLOOR_4_PLUS))

    if fired_count >= 3:
        return max(raw_score, float(CFG.ACCUM_FLOOR_3))

    if fired_count >= 2:
        return max(raw_score, float(CFG.ACCUM_FLOOR_2))

    if fired_medium:
        ev_max = max(float(s["score"]) for s in fired_medium)
        floor = CFG.SAFETY_FLOOR_MEDIUM_BASE + (ev_max - 25.0) * 0.4
        return max(raw_score, min(floor, CFG.SAFETY_FLOOR_MEDIUM_MAX))

    reasons.append("[FLOOR] Safety floor — no forensic anchors detected.")
    return CFG.SAFETY_FLOOR_RESULT


def _morphing_tagged_reasons(video_result: dict, audio_result: dict) -> list[str]:
    out: list[str] = []
    for r in video_result.get("reasons") or []:
        if not isinstance(r, str):
            continue
        if any(
            t in r
            for t in (
                "[FACE-SSIM]",
                "[FACE-WARP]",
                "[MORPH",
                "[META",
                "[AV-SYNC]",
                "[COLOR]",
            )
        ):
            out.append(r)
    for r in audio_result.get("reasons") or []:
        if isinstance(r, str) and ("[PHASE]" in r or "phase discontinu" in r.lower()):
            out.append(r)
    return out[:14]


def build_morphing_modality_result(
    video_result: dict[str, Any], audio_result: dict[str, Any]
) -> dict[str, Any]:
    """
    First-class morphing modality for fusion: score from video pipeline,
    confidence high when non-zero, is_strong when index is definitive.
    """
    mor = float(video_result.get("morphing_score", 0) or 0)
    mor_c = (
        float(CFG.MORPHING_MODALITY_CONFIDENCE)
        if mor > 0
        else 0.22
    )
    spikes = int(
        ((audio_result.get("sub_scores") or {}).get("phase") or {}).get("spike_count", 0)
        or 0
    )
    is_strong = bool(
        mor >= float(CFG.MORPHING_IS_STRONG_THRESHOLD)
        or (
            mor >= float(CFG.MORPHING_STRONG_WITH_PHASE_MIN_SCORE)
            and spikes >= int(CFG.MORPHING_STRONG_PHASE_SPIKE_MIN)
        )
    )
    return {
        "morphing_score": mor,
        "confidence": mor_c,
        "is_strong": is_strong,
        "reasons": _morphing_tagged_reasons(video_result, audio_result),
    }


def compute_final_score(
    image_result: dict,
    audio_result: dict,
    video_result: dict,
    liveness_result: dict,
    morphing_result: Optional[dict[str, Any]] = None,
    url_result: Optional[dict] = None,
) -> dict:
    """Fuse modalities including morphing index; morph anchor can bypass safety floor."""
    morphing_result = morphing_result or {
        "morphing_score": 0.0,
        "confidence": 0.2,
        "is_strong": False,
        "reasons": [],
    }

    img_s = float(image_result.get("score", 0) or 0)
    img_c = float(image_result.get("confidence", 0.5) or 0.5)
    aud_s = float(audio_result.get("score", 0) or 0)
    aud_c = float(audio_result.get("confidence", 0.5) or 0.5)
    vid_s = float(video_result.get("score", 0) or 0)
    vid_c = float(video_result.get("confidence", 0.5) or 0.5)
    mor_s = float(morphing_result.get("morphing_score", 0) or 0)
    mor_c = float(morphing_result.get("confidence", 0.8) or 0.8)

    morphing_is_strong = bool(morphing_result.get("is_strong", False)) or (
        mor_s >= float(CFG.MORPHING_IS_STRONG_THRESHOLD)
    )

    detector_map: dict[str, Any] = {
        "image": image_result,
        "audio": audio_result,
        "video": video_result,
        "morphing": {
            "score": mor_s,
            "confidence": mor_c if mor_s > 0 else 0.2,
            "is_strong": morphing_is_strong,
            "reasons": list(morphing_result.get("reasons") or []),
        },
    }
    if url_result:
        detector_map["url"] = url_result

    raw = fuse_module_results(detector_map)

    penalty, penalty_reason = cross_modal_penalty(
        vid_s, aud_s, img_s, vid_c, aud_c, img_c
    )
    raw = min(raw + penalty, 100.0)

    all_signals = [v for v in detector_map.values() if v]
    raw, liveness_reasons = apply_liveness_reduction(raw, liveness_result, all_signals)

    all_reasons: list[str] = []
    all_reasons.extend(image_result.get("reasons") or [])
    all_reasons.extend(audio_result.get("reasons") or [])
    all_reasons.extend(video_result.get("reasons") or [])
    all_reasons.extend(liveness_reasons)
    if url_result:
        all_reasons.extend(url_result.get("reasons") or [])
    if penalty_reason:
        all_reasons.append(penalty_reason)

    if morphing_is_strong:
        anchored = max(
            raw,
            mor_s * float(CFG.MORPHING_ANCHOR_SCORE_FRACTION),
        )
        final = min(100.0, anchored)
        all_reasons.insert(
            0,
            f"[MORPH-ANCHOR] Strong morphing index ({mor_s:.0f}%) — "
            f"safety floor bypassed; fused score anchored at ≥ "
            f"{mor_s * float(CFG.MORPHING_ANCHOR_SCORE_FRACTION):.0f}%.",
        )
    else:
        final = apply_safety_floor(raw, all_signals, all_reasons)

    final = min(100.0, float(final))

    active_confs: list[float] = []
    for s, c in (
        (img_s, img_c),
        (aud_s, aud_c),
        (vid_s, vid_c),
        (mor_s, mor_c),
    ):
        if s > 0 or c > 0.3:
            active_confs.append(float(c))
    if url_result:
        url_s = float(url_result.get("score", 0) or 0)
        url_c = float(url_result.get("confidence", 0.5) or 0.5)
        if url_s > 0 or url_c > 0.3:
            active_confs.append(url_c)

    overall_confidence = (
        float(sum(active_confs) / len(active_confs)) if active_confs else 0.3
    )

    verdict = (
        "HIGH RISK"
        if final >= CFG.HIGH_RISK_THRESHOLD
        else "MEDIUM RISK"
        if final >= CFG.MEDIUM_RISK_THRESHOLD
        else "LOW RISK"
    )

    url_s_out = float(url_result.get("score", 0) or 0) if url_result else 0.0

    return {
        "score": round(final, 1),
        "verdict": verdict,
        "confidence": round(overall_confidence, 3),
        "reasons": all_reasons,
        "sub_scores": {
            "image": img_s,
            "audio": aud_s,
            "video": vid_s,
            "morphing": mor_s,
            "url": url_s_out,
        },
        "liveness_detected": bool(liveness_result.get("liveness_detected", False)),
        "raw_pre_floor": round(raw, 1),
        "cross_modal_penalty": round(penalty, 1),
        "morphing_score": mor_s,
        "manipulation_score": mor_s,
    }


def _empty_modality() -> dict:
    return {
        "score": 0.0,
        "confidence": 0.0,
        "is_strong": False,
        "reasons": [],
        "sub_scores": {},
    }


def generate_final_verdict_ai(
    all_evidence: dict,
    skip_llm: bool = False,
    model: str | None = None,
    stream: bool = False,
) -> dict:
    """
    Adapter: session `all_evidence` keys Image / Audio / Video / URL → v3.0 fusion.
    """
    if not all_evidence:
        return {
            "final_score": 0,
            "risk_level": "Low",
            "ai_explanation": "No data.",
            "confidence": "0%",
            "key_findings": [],
        }

    img = all_evidence.get("Image") or _empty_modality()
    aud = all_evidence.get("Audio") or _empty_modality()
    vid = all_evidence.get("Video") or _empty_modality()
    url = all_evidence.get("URL")

    liveness = vid.get("liveness") or {
        "liveness_detected": bool(vid.get("metrics", {}).get("liveness_override")),
        "pulse_confirmed": bool(vid.get("metrics", {}).get("pulse_confirmed")),
        "blink_count": int(vid.get("metrics", {}).get("blink_count", 0) or 0),
        "iris_jitter": float(vid.get("metrics", {}).get("iris_jitter", 0.0) or 0.0),
        "confidence": float(vid.get("metrics", {}).get("liveness_confidence", 0.5) or 0.5),
    }

    morphing_mod = build_morphing_modality_result(vid, aud)
    fusion = compute_final_score(img, aud, vid, liveness, morphing_mod, url)
    morphing_score = int(round(float(fusion.get("morphing_score", 0) or 0)))
    final_score = int(round(fusion["score"]))
    verdict_short = (
        "High"
        if final_score >= CFG.HIGH_RISK_THRESHOLD
        else "Medium"
        if final_score >= CFG.MEDIUM_RISK_THRESHOLD
        else "Low"
    )

    overall_conf = int(round(float(fusion.get("confidence", 0.3) or 0.3) * 100))

    use_model = model or CFG.LLM_VERDICT_MODEL

    if skip_llm:
        top = "\n".join(f"- {r}" for r in (fusion.get("reasons") or [])[:12])
        ai_explanation = (
            f"Verdict: {fusion['verdict']} ({final_score}%)\n"
            f"Liveness: {'Confirmed' if fusion.get('liveness_detected') else 'Not detected'}\n"
            f"Morphing index: {morphing_score}%\n"
            f"Findings:\n{top or '- None'}"
        )
    else:
        ai_explanation = llm_generate_explanation(
            all_evidence,
            final_score,
            verdict_short,
            fusion.get("reasons", []),
            model=use_model,
            stream=stream,
        )

    key_findings = (fusion.get("reasons") or [])[:20]

    def _threat_score(ev: Optional[dict]) -> int:
        if not ev:
            return 0
        t = ev.get("threats")
        if isinstance(t, dict):
            return int(t.get("score", 0) or 0)
        return 0

    threat_score = max(
        _threat_score(all_evidence.get("Image")),
        _threat_score(all_evidence.get("Audio")),
        _threat_score(all_evidence.get("Video")),
    )

    return {
        "final_score": final_score,
        "risk_level": verdict_short,
        "confidence": f"{overall_conf}%",
        "key_findings": key_findings,
        "ai_explanation": ai_explanation,
        "ai_generated_score": int(
            max(
                float(img.get("score", 0) or 0),
                float(aud.get("score", 0) or 0),
                float(vid.get("ai_gen_score", 0) or 0)
                or float(vid.get("score", 0) or 0),
            )
        ),
        "manipulation_score": morphing_score,
        "morphing_score": morphing_score,
        "threat_score": int(threat_score),
        "fusion_detail": fusion,
    }
