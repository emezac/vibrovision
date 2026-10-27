#!/usr/bin/env python3
"""A/B comparison of two real clips (e.g. fan without / with a coin on a blade): every
spectral indicator on sliding windows of the floor-referenced housing trajectory, with
bands scaled to the camera frame rate. Implementation: vibrovision/realvideo.py.

    python eval/real_ab.py nocoin.mp4 coin.mp4 --labels "no coin" "coin" [--window 6]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vibrovision.realvideo import AB_KEYS, compare, housing_motion, indicator_windows, summarize  # noqa: E402
from vibrovision.spectral import SpectralConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs=2)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "real_ab.json"))
    a = ap.parse_args()
    res = {}
    for path, lab in zip(a.videos, a.labels):
        m = housing_motion(path)
        rows = indicator_windows(m, a.window, a.step)
        res[lab] = {"file": m["file"], "fps": m["fps"], "windows": rows, "summary": summarize(rows),
                    "fit_band_hz": list(SpectralConfig.for_fps(m["fps"]).fit_band),
                    "tone_displacement_rms_px": m["tone_displacement_rms_px"],
                    "camera_motion_rms_px_per_frame": m["camera_motion_rms_px_per_frame"]}
    A, B = (res[l]["summary"] for l in a.labels)
    sep = compare(A, B)
    print(f"{'indicator':12s} {a.labels[0]:>24s} {a.labels[1]:>24s}   separated?")
    for k in AB_KEYS:
        if A[k] and B[k]:
            f = lambda s: f"{s['median']:.3g} [{s['min']:.3g}, {s['max']:.3g}]"  # noqa: E731
            print(f"{k:12s} {f(A[k]):>24s} {f(B[k]):>24s}   {'yes (no overlap)' if sep[k] else 'no'}")
    with open(a.json, "w") as fh:
        json.dump({"labels": a.labels, "window_s": a.window, "step_s": a.step, **res}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
