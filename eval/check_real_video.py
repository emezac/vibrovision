#!/usr/bin/env python3
"""Pre-flight check for real videos: can this footage support sub-pixel vibrometry?

Reports container facts, raw motion of the most trackable patches, the residual after
OpenCV homography stabilisation (the noise floor any vibration must beat) and the
inspection agent's own setup verdict. Implementation: vibrovision/realvideo.py.

    python eval/check_real_video.py video1.mp4 [video2.mp4 ...] [--out docs/real_video_check.png]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vibrovision.realvideo import plot_preflight, preflight, public  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "real_video_check.png"))
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "real_video_check.json"))
    a = ap.parse_args()
    res = [preflight(p, keep_series=True) for p in a.videos]
    plot_preflight(res, a.out)
    with open(a.json, "w") as fh:
        json.dump([public(r) for r in res], fh, indent=1, default=float)
    for r in res:
        print(f"{r['file']}: {r['fps']:.2f} fps {r['width']}x{r['height']} {r['duration_s']:.1f}s "
              f"{r['bits_per_pixel'] and round(r['bits_per_pixel'], 3)} bit/px | raw motion median "
              f"{np.median(list(r['raw_motion_rms_px'].values())):.1f} px | after stabilisation median "
              f"{np.median(list(r['after_stabilisation_rms_px'].values())):.2f} px | agent: {r['agent_verdict']['problems'] or 'ok'}")


if __name__ == "__main__":
    main()
