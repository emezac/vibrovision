#!/usr/bin/env python3
"""Checks of the image-Jacobian measurement theory (vibrovision/theory.py).

1. sensor-noise estimate vs the true sigma (sensor noise + 8-bit quantisation);
2. predicted vs measured tracker noise: static target (pure noise) and moving parts (noise + linearisation
   error), white floor above 15 Hz vs the predicted floor 2 tr(Cov_IC)/fps;
3. blur trade-off: the theory's noise term vs the total error as the IC pre-blur changes;
4. H.264 (libx264 CRF 12-28): where the white-noise theorem stops holding;
5. the user's real phone clips (static background patches, common mode removed).

    python eval/run_theory.py           (~6 min on 2 vCPU)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2
import numpy as np
from scipy.signal import welch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vibrovision.monitor import REF  # noqa: E402
from vibrovision.pipeline import build_scenario  # noqa: E402
from vibrovision.simulate import generate_displacements, h264_roundtrip, render_frames  # noqa: E402
from vibrovision.theory import estimate_sensor_noise, ic_noise_model  # noqa: E402
from vibrovision.tracking import ICTracker, texture_score, track_sequence  # noqa: E402


def _err(disp, truth, k, n):
    tr = (truth[k][:, :n] - truth[k][:, :1]) if k != REF else 0.0
    e = disp[k][:, :n] - tr
    return e - e.mean(axis=1, keepdims=True)


def _floor(e, fs):
    f, P = welch(e, fs=fs, nperseg=256)
    return float(np.median(P.sum(0)[(f > 15) & (f < 55)]))


def noise_grid():
    rows = []
    for noise, contrast in [(1.0, 1.0), (2.0, 1.0), (4.0, 1.0), (3.0, 0.45), (8.0, 1.0)]:
        sc = build_scenario("healthy", seed=3, cam_noise=noise, contrast=contrast, duration_s=20.0)
        t, truth, shake = generate_displacements(sc)
        boxes = {c.name: c.roi for c in sc.components}
        boxes[REF] = sc.reference_roi
        fr = list(render_frames(sc, truth, shake))
        disp, _, n = track_sequence(fr, boxes, "ic")
        sig = estimate_sensor_noise(fr[:240])["sigma"]
        for k, b in boxes.items():
            e = _err(disp, truth, k, n)
            m = ic_noise_model(fr[0], b, sig, sc.fps)
            rows.append({"noise": noise, "contrast": contrast, "sigma_true": float(np.sqrt(noise ** 2 + 1 / 12)),
                         "sigma_hat": sig, "roi": k, "static": k == REF,
                         "meas_rms": float(np.sqrt(np.sum(e.var(axis=1)))), "pred_rms": m["ic_rms_px"],
                         "crb_rms": m["crb_rms_px"], "floor_ratio": _floor(e, sc.fps) / m["floor_psd"],
                         "lambda_min": m["fisher_lambda_min"]})
        print("noise grid", noise, contrast, flush=True)
    return rows


def blur_sweep():
    rows = []
    sc = build_scenario("healthy", seed=3, cam_noise=2.0, duration_s=10.0)
    t, truth, shake = generate_displacements(sc)
    boxes = {c.name: c.roi for c in sc.components}
    boxes[REF] = sc.reference_roi
    fr = list(render_frames(sc, truth, shake))
    sig = float(np.sqrt(4 + 1 / 12))
    for blur in (0.5, 0.7, 1.0, 1.4, 2.0):
        for k, b in boxes.items():
            trk = ICTracker(fr[0], b, blur=blur)
            d = np.array([trk.track(f)[:2] for f in fr[1:]]).T
            tr = (truth[k][:, 1:len(fr)] - truth[k][:, :1]) if k != REF else 0.0
            e = d - tr
            e = e - e.mean(axis=1, keepdims=True)
            rows.append({"blur": blur, "roi": k, "static": k == REF, "meas_rms": float(np.sqrt(np.sum(e.var(axis=1)))),
                         "pred_noise_rms": ic_noise_model(fr[0], b, sig, sc.fps, blur=blur)["ic_rms_px"]})
        print("blur", blur, flush=True)
    return rows


def codec():
    rows = []
    sc = build_scenario("healthy", seed=3, cam_noise=2.0, duration_s=10.0)
    t, truth, shake = generate_displacements(sc)
    boxes = {c.name: c.roi for c in sc.components}
    boxes[REF] = sc.reference_roi
    with tempfile.TemporaryDirectory() as wd:
        for crf in (None, 12, 18, 23, 28):
            fr = list(render_frames(sc, truth, shake))
            if crf:
                fr = list(h264_roundtrip(iter(fr), sc.fps, crf, wd))
            disp, _, n = track_sequence(fr, boxes, "ic")
            sig = estimate_sensor_noise(fr[:240])["sigma"]
            for k, b in boxes.items():
                e = _err(disp, truth, k, n)
                m = ic_noise_model(fr[0], b, sig, sc.fps)
                rows.append({"crf": crf, "roi": k, "static": k == REF, "sigma_hat": sig,
                             "floor_ratio": _floor(e, sc.fps) / m["floor_psd"],
                             "meas_rms": float(np.sqrt(np.sum(e.var(axis=1)))), "pred_rms": m["ic_rms_px"]})
            print("crf", crf, flush=True)
    return rows


def real_clips():
    from vibrovision.realvideo import load_frames
    rows = []
    for p in ("data/real/fan_nocoin.mp4", "data/real/fan_coin.mp4"):
        path = os.path.join(ROOT, p)
        if not os.path.exists(path):
            continue
        fps, fr = load_frames(path)
        fr = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in fr]
        H, W = fr[0].shape
        c = cv2.HoughCircles(cv2.GaussianBlur(fr[0], (0, 0), 2), cv2.HOUGH_GRADIENT, dp=1.5, minDist=400,
                             param1=120, param2=60, minRadius=int(.25 * H), maxRadius=int(.6 * H))
        cx, cy, r = c[0][0]
        cands = []
        for y in range(10, H - 60, 24):
            for x in range(10, W - 60, 24):
                if np.hypot(x + 24 - cx, y + 24 - cy) > 1.3 * r + 40:
                    cands.append(((x, y, 48, 48), texture_score(fr[0], (x, y, 48, 48))["min_eig_p50"]))
        cands.sort(key=lambda z: -z[1])
        boxes = {}
        for b, _ in cands:
            if all(abs(b[0] - o[0]) > 60 or abs(b[1] - o[1]) > 60 for o in boxes.values()):
                boxes[f"bg{len(boxes)}"] = b
            if len(boxes) == 6:
                break
        disp, _, n = track_sequence(fr, boxes, "ic")
        common = np.median(np.stack(list(disp.values())), axis=0)
        for k, b in boxes.items():
            sig = estimate_sensor_noise(fr[:90], b)["sigma"]
            m = ic_noise_model(fr[0], b, sig, fps)
            e = disp[k] - common
            e = e - e.mean(axis=1, keepdims=True)
            f, P = welch(e, fs=fps, nperseg=64)
            rows.append({"clip": os.path.basename(p), "box": list(b), "sigma_hat": sig, "pred_rms": m["ic_rms_px"],
                         "meas_rms": float(np.sqrt(np.sum(e.var(axis=1)))),
                         "floor_ratio": float(np.median(P.sum(0)[f > 5]) / m["floor_psd"])})
        print("real", p, flush=True)
    return rows


def main():
    res = {"noise_grid": noise_grid(), "blur_sweep": blur_sweep(), "codec": codec(), "real": real_clips()}
    with open(os.path.join(ROOT, "results", "theory_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("results/theory_summary.json written")


if __name__ == "__main__":
    main()
