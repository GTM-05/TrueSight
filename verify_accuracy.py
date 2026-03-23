#!/usr/bin/env python3
"""
Regression / smoke tests for video pipeline + multimodal fusion.
Place optional labeled clips under test_samples/:
  - Showcase_Video_Human.mp4  → expect final score < 25% (low FP)
  - Showcase_Video_AI.mp4     → expect final score >= 60% (catch synthetic)
Any other *.mp4 in test_samples/ is analyzed and reported without pass/fail bands.

Usage:
  python verify_accuracy.py                  # smoke: exit 0 if no crashes
  python verify_accuracy.py --strict-benchmarks  # exit 1 if labeled clips miss targets
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fusion.engine import generate_final_verdict_ai
from modules.video import analyze_video


def _evidence_from_video(vid_res: dict) -> dict:
    """Single-modality video run: mirror minimal audio/image stubs for fusion."""
    aud_s = float((vid_res.get("audio") or {}).get("score", 0) or 0)
    return {
        "Video": vid_res,
        "Audio": {
            "score": aud_s,
            "confidence": 0.55 if aud_s > 5 else 0.25,
            "is_strong": False,
            "reasons": [],
            "sub_scores": {},
        },
        "Image": {
            "score": 0.0,
            "confidence": 0.2,
            "is_strong": False,
            "reasons": [],
            "sub_scores": {},
        },
    }


def test_sample(path: str, label: str) -> tuple[float | None, str]:
    print(f"\n--- {label}: {path} ---")
    if not os.path.isfile(path):
        print("  SKIP: file not found.")
        return None, "missing"

    try:
        vid_res = analyze_video(path, low_resource=False, deep_scan=False)
    except Exception as e:
        print(f"  ERROR: analyze_video crashed: {e}")
        return None, "error"

    print(f"  Video module score: {vid_res.get('score')}%  conf={vid_res.get('confidence')}")
    print(f"  Liveness: {vid_res.get('liveness', {})}")
    m = vid_res.get("metrics") or {}
    print(
        f"  metrics: frames={m.get('num_frames')} rppg_snr={m.get('rppg_snr')} "
        f"audio_score={m.get('audio_score')}"
    )

    verdict = generate_final_verdict_ai(
        _evidence_from_video(vid_res), skip_llm=True
    )
    fs = int(verdict.get("final_score", 0))
    print(f"  Fusion final score: {fs}%  verdict={verdict.get('risk_level')}")
    kf = verdict.get("key_findings") or []
    if kf:
        print(f"  First findings: {kf[:3]}")
    return float(fs), "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description="TrueSight video + fusion checks")
    ap.add_argument(
        "--strict-benchmarks",
        action="store_true",
        help="Exit non-zero when Showcase_* clips miss expected score bands",
    )
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(base, exist_ok=True)

    patterns = sorted(glob.glob(os.path.join(base, "*.mp4")))
    labeled = {
        os.path.join(base, "Showcase_Video_Human.mp4"): (
            "HUMAN (target final < 25%)",
            "human",
        ),
        os.path.join(base, "Showcase_Video_AI.mp4"): ("AI (target final >= 60%)", "ai"),
    }

    to_run: list[tuple[str, str, str | None]] = []
    for p in patterns:
        meta = labeled.get(os.path.abspath(p))
        if meta:
            to_run.append((p, meta[0], meta[1]))
        else:
            to_run.append((p, os.path.basename(p), None))

    if not to_run:
        print(f"No MP4 files in {base}/")
        print("Add clips or run: bash scripts/make_test_videos.sh")
        return 1

    results: dict[str, float | None] = {}
    exit_code = 0
    benchmark_fail = False

    for path, label, kind in to_run:
        score, status = test_sample(path, label)
        results[label] = score
        if status != "ok":
            exit_code = 1
            continue
        if kind == "human" and score is not None and score >= 25:
            print(f"  BENCHMARK MISS: human showcase {score}% (target < 25%)")
            benchmark_fail = True
        elif kind == "human" and score is not None:
            print(f"  BENCHMARK OK: human showcase {score}% < 25%")
        elif kind == "ai" and score is not None and score < 60:
            print(f"  BENCHMARK MISS: AI showcase {score}% (target >= 60%)")
            benchmark_fail = True
        elif kind == "ai" and score is not None:
            print(f"  BENCHMARK OK: AI showcase {score}% >= 60%")

    print("\n=== Summary ===")
    for label, score in results.items():
        print(f"  {label}: {score if score is not None else 'n/a'}")

    if benchmark_fail and args.strict_benchmarks:
        print(
            "\nStrict mode: labeled benchmarks missed targets "
            "(tune CFG / detectors or replace showcase clips)."
        )
        exit_code = 1
    elif benchmark_fail:
        print(
            "\nNote: labeled benchmarks missed targets; "
            "re-run with --strict-benchmarks to fail CI."
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
