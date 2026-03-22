import json
import os
import re
import time
from typing import Any, Optional, Generator

import ollama
import requests

from config import CFG

_TAG_CACHE_DATA: list[str] = []


def _ollama_tag_names() -> list[str]:
    global _TAG_CACHE_DATA
    if _TAG_CACHE_DATA:
        return _TAG_CACHE_DATA
    try:
        base = CFG.OLLAMA_BASE_URL.rstrip("/")
        # Increase timeout slightly to 3s for slower networks
        r = requests.get(f"{base}/api/tags", timeout=3)
        if r.status_code != 200:
            return []
        data = r.json()
        _TAG_CACHE_DATA = [
            m.get("name", "") for m in data.get("models", []) if m.get("name")
        ]
        return _TAG_CACHE_DATA
    except Exception:
        return []


def llm_preload_model(model: Optional[str] = None) -> bool:
    """Trigger Ollama to load the model into memory without generating a full response."""
    try:
        target = model or CFG.LLM_VERDICT_MODEL
        use_model = _resolve_llm_model(target)
        if not use_model:
            return False
        # Empty prompt with 1 token output to trigger load
        ollama.generate(model=use_model, prompt="", options={"num_predict": 1})
        return True
    except Exception:
        return False


def _resolve_llm_model(requested: str | None) -> Optional[str]:
    """Pick first available model: preferred, then fallbacks, then any local tag."""
    tags = _ollama_tag_names()
    if not tags:
        return None

    # 1. Direct match
    if requested and requested in tags:
        return requested

    # 2. Check fallbacks
    for f in CFG.LLM_VERDICT_FALLBACK_MODELS:
        if f in tags:
            return f

    # 3. Just take the first one if it looks relevant
    for t in tags:
        if "qwen" in t.lower() or "phi" in t.lower() or "llama" in t.lower():
            return t

    return tags[0] if tags else None


def _build_narrative_prompt(facts: dict) -> str:
    """
    Plain text prompt — NO JSON, NO structured output requests.
    Qwen2:0.5b cannot reliably produce valid JSON AND coherent text.
    """
    signals = "\n".join(
        f"- {s}"
        for s in (facts["high_risk_signals"] + facts["medium_signals"])[:5]
    ) or "- No significant signals above threshold"

    return (
        f"You are a forensic analyst. Write ONE paragraph (3-4 sentences) "
        f"explaining these forensic findings. Plain English only. "
        f"No JSON. No bullet points. No headers. No markdown.\n\n"
        f"File: {facts['filename']}\n"
        f"Risk score: {facts['score']:.0f}% ({facts['verdict']})\n"
        f"Signals found:\n{signals}\n\n"
        f"Write the paragraph now:"
    )


def _call_ollama_sync(prompt: str, model: str, url: str) -> str:
    """
    Call Ollama with stream=False.
    """
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 80,
                    "top_p": 0.85,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=20,
        )

        if resp.status_code != 200:
            return ""

        data = resp.json()
        raw = data.get("response", "").strip()

        # Strip any JSON that leaked through
        if raw.startswith("{") or raw.startswith("["):
            return ""

        # Strip markdown artifacts
        raw = raw.replace("**", "").replace("##", "").replace("```", "").strip()
        return raw
    except Exception:
        return ""


def _extract_facts(analysis: dict) -> dict:
    # Support both 'score' (base analysis) and 'final_score' (fusion result)
    raw_s = analysis.get("score") or analysis.get("final_score")
    score = float(raw_s or 0)
    verdict = analysis.get("verdict", "UNKNOWN")
    
    # Extract liveness
    liveness = False
    if "liveness" in analysis and isinstance(analysis["liveness"], dict):
        liveness = bool(analysis["liveness"].get("liveness_detected", False))
    elif "liveness_detected" in analysis:
        liveness = bool(analysis.get("liveness_detected"))
        
    sub = analysis.get("sub_scores", {})
    if not sub and "ai_gen_score" in analysis:
        sub = {
            "image": analysis.get("ai_gen_score", 0),
            "audio": analysis.get("audio", {}).get("score", 0) if isinstance(analysis.get("audio"), dict) else 0,
            "video": analysis.get("manip_score", 0)
        }
        
    reasons = analysis.get("reasons", [])
    raw_s = analysis.get("score") or analysis.get("final_score")
    score = float(raw_s or 0)
    morphing = float(analysis.get("morphing_score", 0) or 0)
    filename = analysis.get("filename", "Unknown")

    tag_labels = {
        "ELA": "image re-compression artifact",
        "SRM": "synthetic residual pattern",
        "SPECTRAL": "spectral frequency anomaly",
        "CHROMA": "chromatic alignment anomaly",
        "NOISE": "noise floor irregularity",
        "DCT-GRID": "GAN structural grid",
        "COPY-MOVE": "copy-move forgery",
        "METADATA": "metadata anomaly",
        "MORPH-SSIM": "temporal morphing",
        "MORPH-FLOW": "non-rigid face warping",
        "BLEND": "face boundary blend seam",
        "COLOR": "face colour mismatch",
        "PITCH": "robotic pitch",
        "PHASE": "audio splice",
        "SILENCE": "digital-zero silence",
        "HNR": "clean harmonics (TTS)",
        "ViT": "AI frame synthesis",
        "LIVENESS": "biological liveness",
        "CROSS-MODAL": "modality disagreement",
        "ACCUM": "signal accumulation pattern",
    }

    high_risk = []
    medium = []
    seen = set()

    for r in reasons:
        if not r or not r.startswith("["):
            continue
        try:
            raw_tag = r.split("]")[0].replace("[", "").split(":")[0].strip()
        except Exception:
            continue
            
        label = tag_labels.get(raw_tag, raw_tag.lower())
        if label in seen:
            continue
        seen.add(label)

        if any(
            k in r.lower()
            for k in [
                "strong",
                "definitive",
                "is_strong",
                "phase discontinuities",
                "copy-move",
                "grid",
                "morphing",
                "blend seam",
                "colour mismatch",
            ]
        ):
            high_risk.append(label)
        elif raw_tag not in ("FLOOR", "LIVENESS"):
            medium.append(label)

    if score >= float(CFG.HIGH_RISK_THRESHOLD):
        risk_desc = "strong evidence of manipulation or AI synthesis"
        recommend = "reject and escalate to a human analyst before any use"
    elif score >= float(CFG.MEDIUM_RISK_THRESHOLD):
        risk_desc = "moderate signals suggesting possible manipulation"
        recommend = "apply secondary verification before accepting as authentic"
    else:
        risk_desc = "no definitive forensic anchors"
        recommend = "treat as likely authentic, noting limited biometric data"

    all_signals = high_risk + medium
    if len(all_signals) == 0:
        dominant = "no significant forensic signals"
    elif len(all_signals) == 1:
        dominant = all_signals[0]
    elif len(all_signals) == 2:
        dominant = f"{all_signals[0]} and {all_signals[1]}"
    else:
        dominant = (
            f"{all_signals[0]}, {all_signals[1]}, and {len(all_signals)-2} more"
        )

    return {
        "filename": filename,
        "score": score,
        "verdict": verdict,
        "risk_desc": risk_desc,
        "recommend": recommend,
        "liveness": liveness,
        "img_score": float(sub.get("image", 0) or 0),
        "aud_score": float(sub.get("audio", 0) or 0),
        "vid_score": float(sub.get("video", 0) or 0),
        "mor_score": morphing,
        "high_risk_signals": high_risk,
        "medium_signals": medium,
        "dominant_signals": dominant,
        "signal_count": len(all_signals),
    }


def _build_deterministic_narrative(f: dict) -> str:
    liveness_str = (
        "Biological liveness was confirmed — a real human subject was detected, "
        "though this does not exclude face-swap manipulation of the visual layer."
        if f["liveness"]
        else "No biological liveness signals were confirmed."
    )

    modality_parts = []
    if f["img_score"] > 0:
        modality_parts.append(f"image={f['img_score']:.0f}%")
    if f["aud_score"] > 0:
        modality_parts.append(f"audio={f['aud_score']:.0f}%")
    if f["vid_score"] > 0:
        modality_parts.append(f"video={f['vid_score']:.0f}%")
    if f["mor_score"] > 0:
        modality_parts.append(f"morphing={f['mor_score']:.0f}%")

    modality_str = (
        "Sub-system scores: " + ", ".join(modality_parts) + "."
        if modality_parts
        else "Only video modality was analysed."
    )

    if f["signal_count"] == 0:
        signal_str = (
            "No forensic signals exceeded detection thresholds. "
            "The safety floor was applied to prevent false positives."
        )
    else:
        signal_str = (
            f"The primary forensic indicators were: {f['dominant_signals']}. "
            f"These were evaluated using confidence-weighted max-biased fusion."
        )

    p1 = (
        f"The file '{f['filename']}' received an overall risk score of "
        f"{f['score']:.0f}%, classified as {f['verdict']}. "
        f"This reflects {f['risk_desc']} across the forensic pipeline. "
        f"{liveness_str}"
    )
    p2 = f"{modality_str} {signal_str}"
    p3 = (
        f"Recommendation: {f['recommend']}. "
        f"This is investigative intelligence from TrueSight's offline pipeline — "
        f"not conclusive legal evidence. Human expert review is required for legal use."
    )

    return f"{p1}\n\n{p2}\n\n{p3}"


def generate_reasoning(
    analysis: dict,
    use_ollama: bool = True,
    model: Optional[str] = None,
    ollama_url: Optional[str] = None,
) -> str:
    """
    Returns complete narrative string.
    Fast path: deterministic narrative built instantly from analysis data.
    """
    facts = _extract_facts(analysis)
    base = _build_deterministic_narrative(facts)

    if not use_ollama:
        return base

    target_model = model or CFG.LLM_VERDICT_MODEL
    use_model = _resolve_llm_model(target_model)
    if not use_model:
        return base

    prompt = _build_narrative_prompt(facts)
    qwen_out = _call_ollama_sync(
        prompt, use_model, ollama_url or CFG.OLLAMA_BASE_URL
    )

    if qwen_out and len(qwen_out) > 40:
        paragraphs = base.split("\n\n")
        if len(paragraphs) >= 2:
            paragraphs[1] = qwen_out
            return "\n\n".join(paragraphs)

    return base


def get_narrative_paragraphs(analysis: dict, use_ollama: bool = True) -> list[str]:
    """Returns list of 3 paragraph strings for ReportLab rendering."""
    narrative = generate_reasoning(analysis, use_ollama=use_ollama)
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    while len(paragraphs) < 3:
        paragraphs.append("No additional forensic commentary available.")
    return paragraphs[:3]


def llm_generate_explanation(
    evidence: dict[str, Any],
    score: float,
    verdict_label: str,
    reasons: list[str],
    model: Optional[str] = None,
    stream: bool = False,
) -> Any:
    """
    Compatibility wrapper for existing calls.
    """
    analysis = {
        "score": score,
        "verdict": verdict_label,
        "reasons": reasons,
        "filename": evidence.get("filename", "Unknown"),
        "sub_scores": {
            k: v.get("score", 0) for k, v in evidence.items() if isinstance(v, dict)
        },
    }
    if "liveness" in evidence:
        analysis["liveness"] = evidence["liveness"]

    if stream:
        # Compatibility yield
        res = generate_reasoning(analysis, use_ollama=True, model=model)
        yield {"response": res, "done": True}
    else:
        return generate_reasoning(analysis, use_ollama=True, model=model)
