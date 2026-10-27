#!/usr/bin/env python3
"""Floor-referenced motion of a machine housing + audio cross-check (implementation:
vibrovision/realvideo.py). Works when the camera is not perfectly fixed and when a
free-standing machine walks across the floor.

    python eval/analyze_fan_motion.py nocoin.mp4 coin.mp4 --labels "no coin" "coin" --out docs/fan_motion.png
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vibrovision.realvideo import housing_motion, plot_housing, public  # noqa: E402

analyze = housing_motion          # backwards-compatible name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "fan_motion.png"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "fan_motion.json"))
    a = ap.parse_args()
    labels = a.labels or [""] * len(a.videos)
    res = [housing_motion(p) for p in a.videos]
    plot_housing(res, labels, a.out)
    with open(a.json, "w") as fh:
        json.dump([public(r) | {"label": l} for r, l in zip(res, labels)], fh, indent=1)
    for r, l in zip(res, labels):
        print(f"{r['file']} [{l}]: drift {np.hypot(*r['drift_px']):.1f} px, rotation {r['rotation_deg']:+.2f} deg, "
              f"vibration {r['vibration_rms_px_per_frame']:.2f} px/frame (camera {r['camera_motion_rms_px_per_frame']:.2f}), "
              f"tone {r['tone_hz']:.2f} Hz = {r['tone_displacement_rms_px']:.3f} px, {r['tone_prominence_db_video']:.1f} dB video / "
              f"{r['tone_prominence_db_audio_at_same_hz']} dB audio")


if __name__ == "__main__":
    main()
