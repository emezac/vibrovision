#!/usr/bin/env python3
"""Reproducible throughput benchmark of the VibroVision OpenCV workload.

Run the *same* script on each target and compare the JSON files:

  * x86 baseline   : c7i.large  + pip opencv-python-headless==5.0.0.93
  * Arm baseline   : c7g.large  + pip opencv-python-headless==5.0.0.93
  * Arm + COOL     : c7g.large  + COOL AMI (cv2 provided by COOL)

Nothing is projected or extrapolated: every number in the output was measured
on the machine that wrote the file. The cv2 build string is stored so a
reviewer can verify which OpenCV executed the workload.

    python bench/bench_pipeline.py --label c7g-cool --out results/bench_c7g-cool.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from vibrovision.simulate import default_scene, generate_displacements, render_frames  # noqa: E402
from vibrovision.spectral import SpectralConfig, analyze_window  # noqa: E402
from vibrovision.tracking import ICTracker, make_tracker  # noqa: E402


def instance_type() -> str:
    """EC2 instance type via IMDSv2 (empty outside EC2)."""
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        tok = urllib.request.urlopen(req, timeout=0.5).read().decode()
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": tok})
        return urllib.request.urlopen(req, timeout=0.5).read().decode()
    except Exception:  # noqa: BLE001
        return ""


def timeit(fn, reps: int, warm: int = 20):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median_us": 1e6 * statistics.median(ts), "p95_us": 1e6 * float(np.percentile(ts, 95)),
            "reps": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=platform.machine())
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=1, help="cv2.setNumThreads (1 = per-core cost)")
    ap.add_argument("--seconds", type=float, default=20.0, help="video length for the end-to-end stage")
    a = ap.parse_args()
    cv2.setNumThreads(a.threads)

    sc = default_scene(seed=500, duration_s=a.seconds)
    t, truth, shake = generate_displacements(sc)
    frames = list(render_frames(sc, truth, shake))
    boxes = {c.name: c.roi for c in sc.components}
    boxes["ref"] = sc.reference_roi
    g = frames[1]
    roi = sc.component("Rodamiento_B").roi
    patch = g[roi[1] - 10:roi[1] + roi[3] + 10, roi[0] - 10:roi[0] + roi[2] + 10].astype(np.float32)
    M = np.float32([[1, 0, 0.3], [0, 1, -0.2]])

    prim = {
        "GaussianBlur_64px_f32": timeit(lambda: cv2.GaussianBlur(patch, (0, 0), 1.0), 2000),
        "Sobel_64px_f32": timeit(lambda: cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3), 2000),
        "warpAffine_64px_f32": timeit(lambda: cv2.warpAffine(patch, M, patch.shape[::-1],
                                                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP), 2000),
        "cornerMinEigenVal_frame_f32": timeit(lambda: cv2.cornerMinEigenVal(g.astype(np.float32), 5, 3), 300),
        "cvtColor_resize_frame": timeit(lambda: cv2.resize(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), None, fx=2, fy=2), 300),
    }
    trk = ICTracker(frames[0], roi)
    prim["ICTracker.track_1roi"] = timeit(lambda: trk.track(g), 2000)
    ecc = make_tracker("ecc", frames[0], roi)
    prim["ECCTracker.track_1roi"] = timeit(lambda: ecc.track(g), 500)
    xy = truth["Rodamiento_B"][:, :1440]
    spec = SpectralConfig()
    prim["alpha_window_12s"] = timeit(lambda: analyze_window(xy, spec, 1e-6), 200, warm=5)

    # end-to-end tracking stage: 5 ROIs per frame (4 parts + static reference)
    trackers = {k: make_tracker("ic", frames[0], b) for k, b in boxes.items()}
    t0 = time.perf_counter()
    for fr in frames[1:]:
        for tr in trackers.values():
            tr.track(fr)
    el = time.perf_counter() - t0
    n = len(frames) - 1
    e2e = {"frames": n, "rois": len(boxes), "seconds": el, "frames_per_s": n / el,
           "roi_frames_per_s": n * len(boxes) / el,
           "realtime_factor_at_120fps": (n / el) / 120.0}

    build = cv2.getBuildInformation()
    out = {
        "label": a.label, "instance_type": instance_type(), "machine": platform.machine(),
        "processor": platform.processor(), "python": platform.python_version(),
        "opencv_version": cv2.__version__, "cv2_module": cv2.__file__, "cv_threads": cv2.getNumThreads(),
        "numpy": np.__version__,
        "build_highlights": [ln.strip() for ln in build.splitlines()
                             if any(k in ln for k in ("Baseline", "Dispatched", "HAL", "Kleidi", "Parallel", "Version control"))],
        "primitives": prim, "end_to_end": e2e, "timestamp": time.time(),
    }
    s = json.dumps(out, indent=1)
    print(s)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(s)


if __name__ == "__main__":
    main()
