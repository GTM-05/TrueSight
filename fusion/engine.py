"""
TrueSight — Fusion Engine v3.0 (FIX-1, FIX-2, FIX-6, FIX-8)
Strict execution order: fuse → cross-modal penalty → liveness reduction → safety floor.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

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
        if float(conf_map[k] or 0) > 0.4:
            active[k] = float(score_map[k] or 0)

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
    raw_score: float, liveness: dict[str, Any]
) -> tuple[float, list[str]]:
    """FIX-8: After fusion, before safety floor."""
    if not liveness.get("liveness_detected"):
        return raw_score, []

    pulse = bool(liveness.get("pulse_confirmed", False))
    blinks = int(liveness.get("blink_count", 0) or 0)
    jitter = float(liveness.get("iris_jitter", 0.0) or 0.0)
    conf = float(liveness.get("confidence", 0.5) or 0.5)

    if pulse and blinks >= 2 and jitter > CFG.IRIS_JITTER_MIN:
        factor = CFG.LIVENESS_CONFIRMED_FACTOR
        msg = (
            f"Full liveness confirmed (pulse + {blinks} blinks + iris jitter={jitter:.2f}) — 90% reduction"
        )
    elif pulse or blinks >= 1:
        factor = CFG.LIVENESS_PARTIAL_FACTOR
        msg = f"Partial liveness (pulse={pulse}, blinks={blinks}) — 50% reduction"
    else:
        return raw_score, []

    reduced = max(raw_score * factor * conf, CFG.LIVENESS_MIN_FLOOR)
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

    return float(min(100.0, spatial + meta_m + phase_m))


def apply_safety_floor(
    raw_score: float, signals: list[dict], reasons: list[str]
) -> float:
    """FIX-2: Graduated floor; score ≥ cap → no floor."""
    has_strong = any(
        s.get("is_strong") and float(s.get("confidence", 0) or 0) > 0.5 for s in signals if s
    )
    has_medium = any(
        float(s.get("score", 0) or 0) > 40 and float(s.get("confidence", 0) or 0) > 0.35
        for s in signals
        if s
    )

    if raw_score >= CFG.SAFETY_CAP_SCORE_LIMIT:
        return raw_score

    if has_strong:
        return max(raw_score, CFG.SAFETY_FLOOR_STRONG)

    if has_medium:
        ev_max = max(
            (float(s["score"]) for s in signals if s and float(s.get("score", 0) or 0) > 40),
            default=40.0,
        )
        floor = CFG.SAFETY_FLOOR_MEDIUM_BASE + (ev_max - 40.0) * 0.5
        return max(raw_score, min(floor, CFG.SAFETY_FLOOR_MEDIUM_MAX))

    capped = min(raw_score, CFG.SAFETY_FLOOR_RESULT)
    if raw_score > capped:
        reasons.append(
            "[FLOOR] Safety floor applied — no forensic anchors detected."
        )
    return capped


def compute_final_score(
    image_result: dict,
    audio_result: dict,
    video_result: dict,
    liveness_result: dict,
    url_result: Optional[dict] = None,
) -> dict:
    """Run the 4-step pipeline; returns unified analysis dict."""
    img_s = float(image_result.get("score", 0) or 0)
    img_c = float(image_result.get("confidence", 0.5) or 0.5)
    aud_s = float(audio_result.get("score", 0) or 0)
    aud_c = float(audio_result.get("confidence", 0.5) or 0.5)
    vid_s = float(video_result.get("score", 0) or 0)
    vid_c = float(video_result.get("confidence", 0.5) or 0.5)

    detector_map: dict[str, Any] = {
        "image": image_result,
        "audio": audio_result,
        "video": video_result,
    }
    if url_result:
        detector_map["url"] = url_result

    raw = fuse_module_results(detector_map)

    penalty, penalty_reason = cross_modal_penalty(
        vid_s, aud_s, img_s, vid_c, aud_c, img_c
    )
    raw = min(raw + penalty, 100.0)

    raw, liveness_reasons = apply_liveness_reduction(raw, liveness_result)

    all_reasons: list[str] = []
    all_reasons.extend(image_result.get("reasons") or [])
    all_reasons.extend(audio_result.get("reasons") or [])
    all_reasons.extend(video_result.get("reasons") or [])
    all_reasons.extend(liveness_reasons)
    if url_result:
        all_reasons.extend(url_result.get("reasons") or [])
    if penalty_reason:
        all_reasons.append(penalty_reason)

    all_signals = [v for v in detector_map.values() if v]
    final = apply_safety_floor(raw, all_signals, all_reasons)

    verdict = (
        "HIGH RISK"
        if final >= CFG.HIGH_RISK_THRESHOLD
        else "MEDIUM RISK"
        if final >= CFG.MEDIUM_RISK_THRESHOLD
        else "LOW RISK"
    )

    return {
        "score": round(final, 1),
        "verdict": verdict,
        "reasons": all_reasons,
        "sub_scores": {
            "image": img_s,
            "audio": aud_s,
            "video": vid_s,
            "url": float(url_result.get("score", 0) or 0) if url_result else 0.0,
        },
        "liveness_detected": bool(liveness_result.get("liveness_detected", False)),
        "raw_pre_floor": round(raw, 1),
        "cross_modal_penalty": round(penalty, 1),
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

    fusion = compute_final_score(img, aud, vid, liveness, url)
    morphing_score = int(round(float(vid.get("morphing_score", 0) or 0)))
    fusion["morphing_score"] = float(morphing_score)
    final_score = int(round(fusion["score"]))
    verdict_short = (
        "High"
        if final_score >= CFG.HIGH_RISK_THRESHOLD
        else "Medium"
        if final_score >= CFG.MEDIUM_RISK_THRESHOLD
        else "Low"
    )

    modality_confs = [c for c in [img.get("confidence"), aud.get("confidence"), vid.get("confidence"), (url or {}).get("confidence")] if c is not None]
    overall_conf = int(float(np.mean(modality_confs)) * 100) if modality_confs else 50

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
            fusion,
            final_score,
            verdict_short,
            overall_conf,
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
