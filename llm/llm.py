"""
Ollama-backed forensic narrative. Uses a compact JSON brief so small models
stick to real numbers and avoid inventing filenames or modalities.
"""

from __future__ import annotations

import json
from typing import Any

import ollama

from config import CFG


def _modality_analyzed(evidence: dict, key: str) -> bool:
    """True if this modality was included in the session payload (user ran that analyzer)."""
    return isinstance(evidence.get(key), dict)


def _trim_reasons(reasons: list[str], max_items: int, max_len: int) -> list[str]:
    out: list[str] = []
    for r in (reasons or [])[:max_items]:
        s = (r or "").strip().replace("\n", " ")
        if len(s) > max_len:
            s = s[: max_len - 1] + "…"
        if s:
            out.append(s)
    return out


def _build_forensic_brief(
    evidence: dict[str, Any],
    fusion: dict[str, Any],
    final_score: int,
    verdict: str,
    confidence: int,
) -> dict[str, Any]:
    img = evidence.get("Image") or {}
    aud = evidence.get("Audio") or {}
    vid = evidence.get("Video") or {}
    url = evidence.get("URL") or {}

    img_m = img.get("metrics") if isinstance(img.get("metrics"), dict) else {}
    aud_m = aud.get("metrics") if isinstance(aud.get("metrics"), dict) else {}
    vid_m = vid.get("metrics") if isinstance(vid.get("metrics"), dict) else {}
    vid_ssim = vid.get("ssim") if isinstance(vid.get("ssim"), dict) else {}

    morph = float(vid.get("morphing_score", 0) or vid_m.get("morphing_score", 0) or 0)
    ai_gen = int(vid.get("ai_gen_score", 0) or 0)

    brief: dict[str, Any] = {
        "final_risk_percent": final_score,
        "verdict_bucket": verdict,
        "mean_modality_confidence_percent": confidence,
        "fusion_engine_verdict": fusion.get("verdict"),
        "liveness_detected": bool(fusion.get("liveness_detected")),
        "modalities_with_data": [
            k for k in ("Image", "Audio", "Video", "URL") if _modality_analyzed(evidence, k)
        ],
        "scores": {
            "image_risk_percent": float(img.get("score", 0) or 0),
            "audio_risk_percent": float(aud.get("score", 0) or 0),
            "video_aggregated_risk_percent": float(vid.get("score", 0) or 0),
            "video_ai_synthesis_score_percent": ai_gen,
            "video_morphing_score_percent": round(morph, 1),
            "url_risk_percent": float(url.get("score", 0) or 0) if url else 0.0,
        },
        "confidence_by_modality": {
            "image": float(img.get("confidence", 0) or 0),
            "audio": float(aud.get("confidence", 0) or 0),
            "video": float(vid.get("confidence", 0) or 0),
            "url": float(url.get("confidence", 0) or 0) if url else 0.0,
        },
        "image_signals": {
            "ela_std_dev": img_m.get("ela_std_dev"),
            "ela_mean": img_m.get("ela_mean"),
        },
        "audio_signals": {
            "pitch_std_hz": aud_m.get("pitch_std"),
            "hnr_median_db": aud_m.get("hnr_median"),
        },
        "video_signals": {
            "frames_sampled": vid_m.get("num_frames"),
            "face_roi_ssim_std": vid_ssim.get("ssim_std"),
            "face_roi_ssim_mean": vid_ssim.get("ssim_mean"),
            "face_ssim_anomaly": vid_ssim.get("anomaly"),
            "face_ssim_mode": vid_ssim.get("mode"),
            "face_ssim_score_percent": vid_m.get("face_ssim_score"),
            "face_warp_score_percent": vid_m.get("face_warp_score"),
            "rppg_snr": vid_m.get("rppg_snr"),
            "pulse_confirmed": vid_m.get("pulse_confirmed"),
            "audio_track_score_percent": vid_m.get("audio_score"),
        },
        "mathematical_fusion_subscores": fusion.get("sub_scores") or {},
        "engine_reasons_top": _trim_reasons(
            list(fusion.get("reasons") or []), max_items=16, max_len=200
        ),
    }
    return brief


def _fallback_narrative(
    fusion: dict[str, Any], final_score: int, verdict: str
) -> str:
    reasons = _trim_reasons(list(fusion.get("reasons") or []), 10, 220)
    bullets = "\n".join(f"- {r}" for r in reasons) if reasons else "- No tagged findings."
    return (
        f"Executive summary: Fusion engine reports {verdict} overall risk at {final_score}% "
        f"(verdict label: {fusion.get('verdict', 'N/A')}).\n\n"
        f"Key quantitative findings (do not contradict these numbers):\n{bullets}\n\n"
        f"Note: The LLM narrative was unavailable or too short; this block is deterministic "
        f"from the fusion engine only."
    )


def llm_generate_explanation(
    evidence: dict[str, Any],
    fusion: dict[str, Any],
    final_score: int,
    verdict: str,
    confidence: int,
    model: str | None = None,
    stream: bool = False,
) -> str | Any:
    brief = _build_forensic_brief(evidence, fusion, final_score, verdict, confidence)
    brief_json = json.dumps(brief, indent=2, ensure_ascii=False)

    system = (
        "You are a senior multimedia forensic analyst writing for investigators. "
        "Use ONLY facts from the JSON block. Do not invent file names, URLs, people, or "
        "modalities that are absent from modalities_with_data. If a field is null or a "
        "modality is missing, say it was not analyzed. Do not contradict final_risk_percent "
        "or any numeric score fields in the JSON. Write clear English; avoid filler."
    )

    prompt = f"""FORENSIC_CASE_BRIEF (JSON — authoritative):
{brief_json}

TASK — produce four short sections with headers exactly as below:

1) EXECUTIVE SUMMARY (2 sentences)
2) PER-MODALITY SIGNALS — only for modalities listed in modalities_with_data; cite numbers from scores / *_signals
3) FUSION — why final_risk_percent and verdict_bucket follow from the combined evidence (1 short paragraph)
4) LIMITATIONS — what was not run or what remains uncertain (1 short paragraph)

Keep total length under 350 words. No markdown tables."""

    opts = {
        "temperature": float(CFG.LLM_VERDICT_TEMPERATURE),
        "top_p": float(CFG.LLM_VERDICT_TOP_P),
        "num_predict": int(CFG.LLM_VERDICT_NUM_PREDICT),
    }
    use_model = model or CFG.LLM_VERDICT_MODEL

    def _postprocess(text: str) -> str:
        t = (text or "").strip()
        if len(t) < int(CFG.LLM_VERDICT_MIN_CHARS):
            return _fallback_narrative(fusion, final_score, verdict) + "\n\n---\n(LLM output was brief or empty; appended engine summary.)"
        return t

    try:
        if stream:
            return ollama.generate(
                model=use_model,
                prompt=prompt,
                system=system,
                stream=True,
                options=opts,
            )

        response = ollama.generate(
            model=use_model,
            prompt=prompt,
            system=system,
            options=opts,
        )
        raw = (response.get("response") or "").strip()
        return _postprocess(raw)
    except Exception as exc:
        return (
            _fallback_narrative(fusion, final_score, verdict)
            + f"\n\n---\n[LLM error: {exc}]"
        )
