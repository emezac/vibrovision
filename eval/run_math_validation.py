#!/usr/bin/env python3
"""Recompute mathematical checks and optionally both bundled real clips."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
from vibrovision.validation import run_checks
from vibrovision.realvideo import compare_clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--bootstrap-reps", type=int, default=24)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "math_validation_v3.json")
    args = ap.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(2026)
    result = run_checks(args.bootstrap_reps)
    if args.real:
        result["real_video"] = compare_clips([ROOT / "data/real/fan_nocoin.mp4", ROOT / "data/real/fan_coin.mp4"],
                                           out_figure=args.out.with_suffix(".png"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"out": str(args.out), "algebra": result["algebra"]["passed"],
                      "synthetic_video": result["synthetic_video"]["passed"], "real_video": "real_video" in result}))
    if not result["algebra"]["passed"] or not result["synthetic_video"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
