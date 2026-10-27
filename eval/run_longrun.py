#!/usr/bin/env python3
"""Long-horizon false-alarm study (hours of healthy operation) and severity at detection.

The 3-minute end-to-end runs can only show ~5 % false alarms per ROI per 2 min,
which is useless for a plant. Here thresholds are set for realistic rates
(1 false alarm per ROI per day / per week) using hours of simulated monitoring
through the *validated surrogate* (true motion + measured tracker-error spectrum,
vibrovision/surrogate.py), and we report how far a fault must develop before it
is detected at those thresholds.

Conditions:  stationary | operating (speed +/-3 %, load +/-25 %, minutes)
             | operating + alpha coupled to load (+/-0.05)  (sensitivity test)
Baselines:   plain CUSUM on each indicator, and a condition-aware alpha baseline
             (alpha regressed on log RMS and shaft-tone frequency during calibration).

    python eval/run_longrun.py [--hours 2 --runs 5]   (~15 min on 2 vCPU)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vibrovision.detect import SD_FLOOR  # noqa: E402
from vibrovision.monitor import REF  # noqa: E402
from vibrovision.pipeline import build_scenario, sliding_indicators  # noqa: E402
from vibrovision.simulate import (FaultSpec, default_scene, generate_displacements,  # noqa: E402
                                  render_frames, severity_profile)
from vibrovision.spectral import DIRECTION, INDICATORS, SpectralConfig, analyze_window, noise_floor_from_reference, psd  # noqa: E402
from vibrovision.surrogate import load_model, surrogate_measurements  # noqa: E402
from vibrovision.tracking import texture_score  # noqa: E402

WIN, STEP = 12.0, 2.0
CONDITIONS = {
    "stationary": {},
    "operating": {"speed_var": 0.03, "load_var": 0.25},
    "operating_alpha_coupled": {"speed_var": 0.03, "load_var": 0.25, "alpha_load_coupling": 0.05},
}
H_GRID = np.round(np.arange(1.0, 30.01, 0.5), 2)
MODEL = None


def _init(model_path):
    global MODEL
    import cv2
    cv2.setNumThreads(1)
    MODEL = load_model(model_path)


def _energies(sc, truth, shake):
    fr = next(render_frames(sc, truth, shake, 0, 1))
    boxes = {c.name: c.roi for c in sc.components}
    boxes[REF] = sc.reference_roi
    return {k: texture_score(fr, b)["grad_energy"] * b[2] * b[3] for k, b in boxes.items()}


def shaft_tone_hz(xy, cfg, band=(20.0, 30.0)):
    f, P = psd(xy - xy.mean(axis=1, keepdims=True), cfg)
    m = (f >= band[0]) & (f <= band[1])
    i = int(np.argmax(np.where(m, P, 0)))
    if 0 < i < len(P) - 1:          # parabolic interpolation of the peak
        a, b, c = np.log(P[i - 1: i + 2])
        return float(f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))
    return float(f[i])


def windows(disp, E, fs, rois):
    cfg = SpectralConfig(fs=fs)
    w, s = int(WIN * fs), int(STEP * fs)
    n = disp[REF].shape[1]
    out = {r: {k: [] for k in INDICATORS + ("tone_hz",)} for r in rois}
    times = []
    for i in range(0, n - w + 1, s):
        times.append((i + w / 2) / fs)
        nref = noise_floor_from_reference(disp[REF][:, i:i + w], cfg)
        for r in rois:
            xy = disp[r][:, i:i + w]
            res = analyze_window(xy, cfg, nref * E[REF] / E[r])
            for k in INDICATORS:
                out[r][k].append(res[k])
            out[r]["tone_hz"].append(shaft_tone_hz(xy, cfg))
    return np.asarray(times), {r: {k: np.asarray(v) for k, v in d.items()} for r, d in out.items()}


def job_healthy(args):
    cond, seed, hours = args
    sc = default_scene(fault=False, seed=seed, duration_s=hours * 3600.0, **CONDITIONS[cond])
    t, truth, shake = generate_displacements(sc)
    E = _energies(sc, truth, shake)
    disp = surrogate_measurements(truth, MODEL, np.random.default_rng(seed + 991))
    del truth
    times, ind = windows(disp, E, sc.fps, [c.name for c in sc.components])
    return {"cond": cond, "seed": seed, "times": times, "ind": ind}


def job_fault(args):
    cond, seed, ring = args
    fault = FaultSpec(onset_s=720.0, full_s=1800.0, ring_hz=ring)
    sc = default_scene(fault=True, seed=seed, duration_s=1800.0, **CONDITIONS[cond])
    sc.component("Rodamiento_B").fault = fault
    t, truth, shake = generate_displacements(sc)
    E = _energies(sc, truth, shake)
    disp = surrogate_measurements(truth, MODEL, np.random.default_rng(seed + 992))
    times, ind = windows(disp, E, sc.fps, ["Rodamiento_B"])
    bb = default_scene().component("Rodamiento_B").broadband_rms_px
    return {"cond": cond, "seed": seed, "ring": ring, "times": times, "ind": ind,
            "impact_rms_full_px": fault.impact_rms_px, "healthy_rms_px": bb,
            "fault": {"onset_s": fault.onset_s, "full_s": fault.full_s, "growth": fault.growth}}


# ----------------------------------------------------------------------------- statistics
def zscores(d, key, cal, aware=False):
    """Baseline from calibration windows; optional condition-aware regression for alpha."""
    v = d[key]
    if not aware:
        mu = np.nanmedian(v[cal])
        sd = max(np.nanstd(v[cal]), SD_FLOOR[key])
        z = (v - mu) / sd
    else:
        X = np.column_stack([np.ones_like(v), np.log(d["rms"]), d["tone_hz"] / np.nanmean(d["tone_hz"][cal]) - 1.0])
        ok = cal & np.isfinite(v)
        beta, *_ = np.linalg.lstsq(X[ok], v[ok], rcond=None)
        res = v - X @ beta
        sd = max(np.nanstd(res[ok]), SD_FLOOR[key] * 0.5)
        z = res / sd
    return -z if DIRECTION[key] == "down" else z


def alarm_counts(z, start, weight, k=0.5):
    """Alarms for every h in H_GRID with renewal (reset after each alarm), from
    index `start` on. Vectorised over thresholds."""
    g = np.zeros(len(H_GRID))
    c = np.zeros(len(H_GRID), int)
    for x in z[start:]:
        if np.isfinite(x):
            g = np.maximum(0.0, g + weight * (x - k))
            hit = g > H_GRID
            c += hit
            g[hit] = 0.0
    return c


def first_alarm_t(z, times, start, weight, h, k=0.5):
    g = 0.0
    for i in range(start, len(z)):
        if np.isfinite(z[i]):
            g = max(0.0, g + weight * (z[i] - k))
            if g > h:
                return float(times[i])
    return None


def h_for_rate(counts, hours, rate_per_hour):
    """Threshold h for a target false-alarm rate.

    1) 'certified': smallest h whose Poisson 95 % upper bound on the rate is <= target
       (with zero alarms in T hours the bound is ~3/T, so this needs enough hours);
    2) otherwise 'extrapolated': log-linear fit of the observed rate vs h (CUSUM's
       average run length grows ~exponentially with h), using points with >= 3 alarms.
    """
    from scipy.stats import chi2
    upper = np.array([chi2.ppf(0.95, 2 * (c + 1)) / 2.0 / hours for c in counts])
    ok = [h for h, u in zip(H_GRID, upper) if u <= rate_per_hour]
    if ok:
        return float(ok[0]), {"method": "certified"}
    pts = [(h, np.log(c / hours)) for h, c in zip(H_GRID, counts) if c >= 3]
    pts = pts[-6:]                       # fit the tail only (ARL growth is exponential there)
    if len(pts) >= 3:
        hh, ll = np.array(pts).T
        slope, icpt = np.polyfit(hh, ll, 1)
        if slope < 0:
            return float((np.log(rate_per_hour) - icpt) / slope), {"method": "extrapolated"}
    return None, {"method": "insufficient data"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--fault-runs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    T0 = time.time()
    model_path = os.path.join(ROOT, "results", "tracker_error_model.json")
    hjobs = [(c, 5000 + i, a.hours) for c in CONDITIONS for i in range(a.runs)]
    fjobs = [(c, 6000 + i, [30.0, 41.0][i % 2]) for c in CONDITIONS for i in range(a.fault_runs)]
    pkl = os.path.join(ROOT, "results", "longrun_windows.pkl")
    if a.analyze_only:
        with open(pkl, "rb") as fh:
            dd = pickle.load(fh)
        healthy, faults = dd["healthy"], dd["faults"]
        missing = [j for j in fjobs if not any(f["cond"] == j[0] and f["seed"] == j[1] for f in faults)]
        if missing:                      # e.g. fault runs for a condition added later
            with Pool(a.workers, initializer=_init, initargs=(model_path,)) as pool:
                faults += pool.map(job_fault, missing)
            with open(pkl, "wb") as fh:
                pickle.dump({"healthy": healthy, "faults": faults}, fh)
    else:
        with Pool(a.workers, initializer=_init, initargs=(model_path,)) as pool:
            healthy = pool.map(job_healthy, hjobs)
            faults = pool.map(job_fault, fjobs)
        with open(pkl, "wb") as fh:
            pickle.dump({"healthy": healthy, "faults": faults}, fh)

    weight = STEP / WIN
    summary = {"config": {"hours_per_run": float(np.mean([r["times"][-1] for r in healthy]) / 3600),
                          "runs": len(healthy) // len(CONDITIONS), "window_s": WIN, "step_s": STEP,
                          "conditions": CONDITIONS}, "far": {}, "detect": {}}
    variants = [(ind, False) for ind in ("alpha", "band_ratio", "hf_rms", "rms")] + [("alpha", True)]
    thresholds = {}
    for cond in CONDITIONS:
        runs = [r for r in healthy if r["cond"] == cond]
        for cal_s in (60.0, 600.0):
            for ind, aware in variants:
                name = f"{ind}{'+cond' if aware else ''}"
                counts = np.zeros(len(H_GRID), int)
                hours = 0.0
                for r in runs:
                    cal = r["times"] + WIN / 2 <= cal_s
                    start = int(cal.sum())
                    if start >= len(r["times"]) - 1:
                        continue
                    for roi, d in r["ind"].items():
                        counts += alarm_counts(zscores(d, ind, cal, aware), start, weight)
                        hours += (r["times"][-1] - r["times"][start]) / 3600.0
                h_day, info_d = h_for_rate(counts, hours, 1 / 24)
                h_week, info_w = h_for_rate(counts, hours, 1 / 168)
                if h_day is not None and h_week is not None and h_week < h_day:
                    h_week = h_day     # a rarer target can never need a lower threshold
                key = f"{cond}|cal{int(cal_s)}|{name}"
                summary["far"][key] = {"roi_hours": hours, "alarms_per_hour_at_h": dict(zip(map(float, H_GRID), (counts / hours).tolist())),
                                       "h_1_per_day": h_day, "h_1_per_week": h_week,
                                       "method_day": info_d["method"], "method_week": info_w["method"],
                                       "fa_per_hour_at_h6": float(counts[list(H_GRID).index(6.0)] / hours)}
                thresholds[key] = (h_day, h_week)

    # severity at detection (calibration 600 s, thresholds from the matching condition)
    for cond in sorted({f["cond"] for f in faults}, key=list(CONDITIONS).index):
        for ind, aware in variants:
            name = f"{ind}{'+cond' if aware else ''}"
            h_day, h_week = thresholds[f"{cond}|cal600|{name}"]
            res = {}
            for tag, h in (("1_per_day", h_day), ("1_per_week", h_week)):
                sev, det = [], 0
                for r in [f for f in faults if f["cond"] == cond]:
                    d = r["ind"]["Rodamiento_B"]
                    cal = r["times"] + WIN / 2 <= 600.0
                    z = zscores(d, ind, cal, aware)
                    ta = None if h is None else first_alarm_t(z, r["times"], int(cal.sum()), weight, h)
                    if ta is not None and ta < r["fault"]["onset_s"]:
                        ta = None          # alarm before onset = false alarm, not a detection
                    if ta is not None:
                        det += 1
                        fs = FaultSpec(onset_s=r["fault"]["onset_s"], full_s=r["fault"]["full_s"], growth=r["fault"]["growth"])
                        s_ = float(severity_profile(np.array([ta]), fs)[0])
                        sev.append(s_ * r["impact_rms_full_px"])
                n = len([f for f in faults if f["cond"] == cond])
                res[tag] = {"h": h, "detected": det, "n": n,
                            "median_impact_rms_px_at_detection": float(np.median(sev)) if sev else None,
                            "impact_to_healthy_rms_pct": float(100 * np.median(sev) / faults[0]["healthy_rms_px"]) if sev else None}
            summary["detect"][f"{cond}|{name}"] = res
    summary["wall_s"] = time.time() - T0
    with open(os.path.join(ROOT, "results", "longrun_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for k, v in summary["far"].items():
        print(f"{k:45s} FA/h@h6={v['fa_per_hour_at_h6']:.3f}  h(1/day)={v['h_1_per_day']} [{v['method_day']}]  h(1/week)={v['h_1_per_week']} [{v['method_week']}]")
    for k, v in summary["detect"].items():
        print(k, json.dumps(v))
    print(f"done in {summary['wall_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
