#!/usr/bin/env python3
"""Camera-artefact robustness, end-to-end (render -> [H.264] -> OpenCV -> indicators).

Conditions: global shutter baseline, auto-exposure (with and without the
photometric normalisation of the IC tracker), rolling shutter, H.264 at three
quality levels (libx264 CRF 18/23/28, yuv420p, like a phone or NVR), and a
combined "phone-like" condition.

    python eval/run_camera.py [--runs 3]       (~30-40 min on 2 vCPU)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vibrovision.detect import SD_FLOOR, alarm_time_for_h, cusum_trace  # noqa: E402
from vibrovision.monitor import REF  # noqa: E402
from vibrovision.pipeline import build_scenario, sliding_indicators  # noqa: E402
from vibrovision.simulate import generate_displacements, h264_roundtrip, render_frames  # noqa: E402
from vibrovision.tracking import ICTracker, fill_nans, texture_score, track_sequence  # noqa: E402

CONDITIONS = {
    "baseline": {},
    "auto_exposure": {"scene": {"exposure_var": 0.15}},
    "auto_exposure_no_photometric": {"scene": {"exposure_var": 0.15}, "photometric": False},
    "rolling_shutter_8ms": {"scene": {"rolling_shutter_s": 0.008}},
    "h264_crf18": {"crf": 18},
    "h264_crf23": {"crf": 23},
    "h264_crf28": {"crf": 28},
    "phone_like": {"scene": {"rolling_shutter_s": 0.006, "exposure_var": 0.08}, "crf": 20},
}
H_ALPHA = 3.0          # alpha threshold that gave <= 5 % FA per ROI in the 3-min main evaluation


def _init():
    import cv2
    cv2.setNumThreads(1)


def job(args):
    cond, kind, seed = args
    spec = CONDITIONS[cond]
    sc = build_scenario(kind, seed)
    sc = replace(sc, **spec.get("scene", {}))
    t, truth, shake = generate_displacements(sc)
    frames = render_frames(sc, truth, shake)
    if "crf" in spec:
        frames = h264_roundtrip(frames, sc.fps, spec["crf"], tempfile.gettempdir())
    frames = list(frames)
    boxes = {c.name: c.roi for c in sc.components}
    boxes[REF] = sc.reference_roi
    if spec.get("photometric", True):
        disp, _, _ = track_sequence(frames, boxes, "ic")
    else:
        trs = {k: ICTracker(frames[0], b, photometric=False) for k, b in boxes.items()}
        dd = {k: [(0.0, 0.0)] + [tr.track(f)[:2] for f in frames[1:]] for k, tr in trs.items()}
        disp = {k: fill_nans(np.asarray(v, float).T) for k, v in dd.items()}
    E = {k: texture_score(frames[0], b)["grad_energy"] * b[2] * b[3] for k, b in boxes.items()}
    n = min(d.shape[1] for d in disp.values())
    disp = {k: v[:, :n] for k, v in disp.items()}
    err = float(np.mean([np.sqrt(np.mean((disp[k] - (truth[k][:, :n] - truth[k][:, :1])) ** 2))
                         for k in boxes if k != REF]))
    ts, ind = sliding_indicators(disp, sc.fps, 12.0, 1.0, energies=E)
    _, ind_t = sliding_indicators({k: v[:, :n] for k, v in truth.items()}, sc.fps, 12.0, 1.0)
    return {"cond": cond, "kind": kind, "seed": seed, "times": ts, "ind": ind, "ind_truth": ind_t,
            "track_err_px": err, "onset": None if kind == "healthy" else sc.component("Rodamiento_B").fault.onset_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--only", nargs="*", default=None, help="re-run only these conditions and merge")
    a = ap.parse_args()
    T0 = time.time()
    conds = a.only or list(CONDITIONS)
    jobs = [(c, k, 7000 + i) for c in conds for k in ("healthy", "fault") for i in range(a.runs)]
    # longest jobs first (rolling shutter renders row bands)
    jobs.sort(key=lambda j: 0 if "rolling" in j[0] or "phone" in j[0] else 1)
    with Pool(a.workers, initializer=_init) as pool:
        res = pool.map(job, jobs)
    path = os.path.join(ROOT, "results", "camera_summary.json")
    out = json.load(open(path)) if (a.only and os.path.exists(path)) else {}
    for cond in conds:
        rr = [r for r in res if r["cond"] == cond]
        bias, gmax, det, delays, errs = [], [], 0, [], []
        nf = 0
        for r in rr:
            errs.append(r["track_err_px"])
            cal = r["times"] + 6 <= 60
            for roi, d in r["ind"].items():
                g = cusum_trace(d["alpha"], cal, "down", 0.5, SD_FLOOR["alpha"], 1 / 12)
                if roi == "Rodamiento_B" and r["onset"] is not None:
                    nf += 1
                    ta = alarm_time_for_h(r["times"], g, H_ALPHA)
                    if ta is not None and ta >= r["onset"]:
                        det += 1
                        delays.append(ta - r["onset"])
                else:
                    gmax.append(g.max())
                    bias.append(np.nanmean(d["alpha"][cal]) - np.nanmean(r["ind_truth"][roi]["alpha"][cal]))
        out[cond] = {"track_err_px": float(np.mean(errs)), "alpha_bias": float(np.mean(bias)),
                     "alpha_bias_sd": float(np.std(bias)), "healthy_series": len(gmax),
                     "false_alarms_at_h3": int(np.sum(np.array(gmax) > H_ALPHA)),
                     "faults": nf, "detected": det, "median_delay_s": float(np.median(delays)) if delays else None}
        print(f"{cond:30s} err={out[cond]['track_err_px']:.4f}px bias={out[cond]['alpha_bias']:+.3f} "
              f"FA={out[cond]['false_alarms_at_h3']}/{len(gmax)} det={det}/{nf} delay={out[cond]['median_delay_s']}")
    out["_meta"] = {"runs": a.runs, "h_alpha": H_ALPHA, "wall_s": time.time() - T0}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
