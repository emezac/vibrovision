#!/usr/bin/env python3
"""Structural-Jacobian study: identify M^-1 K from video and classify faults by what they change.

Bench: 4 coupled parts (mounts K_g + coupling Laplacian L_W, eq. (1) in vibrovision/structure.py),
rendered to 120 fps video, tracked with OpenCV (IC tracker), identified with SSI-cov over 40 s
windows (step 10 s), both image axes. Five conditions, each over several seeds:

  healthy    nothing changes
  mount      Rodamiento_B mount stiffness -> 30 %        (Jacobian change, local)
  joint      Caja_Engranajes-Rodamiento_A coupling -> 20 % (Jacobian change, a cut)
  imbalance  1x rotation force on Motor x4              (forcing change, Jacobian unchanged)
  bearing    impacts on Rodamiento_B (as in the main bench) (forcing change, Jacobian unchanged)

Decision rule (thresholds are set from separate healthy calibration seeds, not tuned on faults):
  structural  if some mount or coupling of M^-1 K moved beyond the healthy range  -> where
  forcing     elif the tonal prominence at a rotation harmonic grew               -> imbalance-like
  impacts     elif alpha fell (broadband shape change)                            -> bearing-like
  healthy     otherwise

The spectral indicators use the noise floor predicted by the image-Jacobian theory
(vibrovision/theory.py) instead of the static reference target.

    python eval/run_structure.py [--seeds 3] [--calib-seeds 3] [--analyze-only]
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

from vibrovision.monitor import REF  # noqa: E402
from vibrovision.simulate import FaultSpec, default_scene, render_frames, severity_profile  # noqa: E402
from vibrovision.spectral import SpectralConfig, analyze_window, tone_power_db  # noqa: E402
from vibrovision.structure import StructFault, StructuralWatch, StructureSpec, simulate_structure  # noqa: E402
from vibrovision.theory import estimate_sensor_noise, ic_noise_model  # noqa: E402
from vibrovision.tracking import track_sequence  # noqa: E402

FPS, DUR, WIN, STEP, CAL_END = 120.0, 180.0, 40.0, 10.0, 60.0
TONES = (24.6, 49.2, 36.9)
CONDITIONS = {
    "healthy": dict(),
    "mount": dict(faults=(StructFault("mount", ("Rodamiento_B",), 0.3),)),
    "joint": dict(faults=(StructFault("joint", ("Caja_Engranajes", "Rodamiento_A"), 0.2),)),
    "imbalance": dict(faults=(StructFault("imbalance", ("Motor",), 4.0),)),
    "bearing": dict(bearing=("Rodamiento_B", FaultSpec(onset_s=60.0, full_s=160.0))),
}
TRUE_CLASS = {"healthy": "healthy", "mount": "structural", "joint": "structural", "imbalance": "forcing",
              "bearing": "impacts"}
TRUE_WHERE = {"mount": "Rodamiento_B", "joint": "Caja_Engranajes|Rodamiento_A"}
OUT = os.path.join(ROOT, "results")


def run_one(args):
    cond, seed = args
    spec = StructureSpec(**CONDITIONS[cond])
    t, truth, _ = simulate_structure(spec, FPS, DUR, seed=seed)
    sc = default_scene(fault=False, duration_s=DUR, seed=seed)
    boxes = {c.name: c.roi for c in sc.components}
    boxes[REF] = sc.reference_roi
    shake = np.zeros((2, len(t)))
    first = list(render_frames(sc, truth, shake, 0, 240))
    sigma = estimate_sensor_noise(first)["sigma"]
    floors = {k: ic_noise_model(first[0], b, sigma, FPS)["floor_psd"] for k, b in boxes.items()}
    disp, _, n = track_sequence(render_frames(sc, truth, shake), boxes, "ic")
    cfg = SpectralConfig.for_fps(FPS)
    watch = StructuralWatch(spec.names, FPS, TONES)          # the agent's component, thresholds applied offline
    rows = []
    for s0 in np.arange(0.0, DUR - WIN + 1e-9, STEP):
        w = slice(int(s0 * FPS), int((s0 + WIN) * FPS))
        d = {k: v[:, w] for k, v in disp.items()}
        ev = watch.update(d, float(s0 + WIN))
        row = {"t": float(s0 + WIN), "watch": {k: v for k, v in ev.items() if k != "change"}}
        if ev.get("ok"):
            row["mount_err"] = [float(m / tm - 1) for ax, ms in ev["mount_abs"].items()
                                for m, tm in zip(ms, spec.truth({}, ax)["mount"])]
        row["spectral"] = {}
        for k in spec.names:     # last 12 s of the window, like the main monitor
            xy = d[k][:, -int(12 * FPS):]
            a = analyze_window(xy, cfg, floors[k])
            row["spectral"][k] = {i: a[i] for i in ("alpha", "tonal_db", "rms")}
            row["spectral"][k]["tone_db"] = tone_power_db(xy, cfg, TONES)
        rows.append(row)
    sev = {fi: float(severity_profile(np.array([DUR]), FaultSpec(onset_s=f.onset_s, full_s=f.full_s, growth=0))[0])
           for fi, f in enumerate(spec.faults)}
    true_final = [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in spec.truth(sev, ax).items()} for ax in (0, 1)]
    true_start = [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in spec.truth({}, ax).items()} for ax in (0, 1)]
    return {"cond": cond, "seed": seed, "sigma": sigma, "floors": floors, "rows": rows,
            "truth_start": true_start, "truth_final": true_final, "names": list(spec.names)}


# ----------------------------------------------------------------------------- analysis
def features(run):
    """Per post-calibration window: relative change of each mount / coupling of M^-1 K (from the agent's
    StructuralWatch), change of the rotation-tone power (dB) and of alpha per part vs calibration."""
    names = run["names"]
    cal = [r for r in run["rows"] if r["t"] <= CAL_END]
    sp0 = {k: {i: np.nanmedian([r["spectral"][k][i] for r in cal]) for i in ("alpha", "tone_db")} for k in names}
    out = []
    for r in run["rows"]:
        w = r["watch"]
        f = {"t": r["t"], "ok": bool(w.get("ok")) and w.get("phase") == "watching"}
        if f["ok"]:
            f["mount_rel"] = [w["mount_rel"][k] for k in names]
            f["coupling_rel"] = w["coupling_rel"]
            f["graph_gap"] = w["graph_gap"]
            f["stability_gap"] = w["stability_gap"]
            f["fiedler_cut"] = w["fiedler_cut"]
            f["extra_modes_hz"] = w.get("extra_modes_hz")
        if w.get("ok"):
            f["graph_gap_any"] = w["graph_gap"]
            f["stability_gap_any"] = w["stability_gap"]
        f["tonal_delta"] = {k: r["spectral"][k]["tone_db"] - sp0[k]["tone_db"] for k in names}
        f["alpha_delta"] = {k: r["spectral"][k]["alpha"] - sp0[k]["alpha"] for k in names}
        out.append(f)
    return out


def classify(f, thr, names):
    if f["ok"]:
        mz = np.abs(f["mount_rel"]) / thr["mount"]
        cz = {e: abs(v) / thr["coupling"] for e, v in f["coupling_rel"].items()}
        e_best = max(cz, key=cz.get)
        if max(mz.max(), cz[e_best]) > 1.0:
            if cz[e_best] > mz.max():
                return "structural", e_best
            return "structural", names[int(np.argmax(mz))]
    td = {k: v for k, v in f["tonal_delta"].items() if np.isfinite(v)}
    if td and max(td.values()) > thr["tonal_db"]:
        return "forcing", max(td, key=td.get)
    ad = {k: v for k, v in f["alpha_delta"].items() if np.isfinite(v)}
    if ad and min(ad.values()) < -thr["alpha"]:
        return "impacts", min(ad, key=ad.get)
    return "healthy", None


def thresholds(calib_feats, margin=1.5):
    post = [f for fs in calib_feats for f in fs if f["t"] > CAL_END]
    mount = max(np.max(np.abs(f["mount_rel"])) for f in post if f["ok"])
    coup = max(max(abs(v) for v in f["coupling_rel"].values()) for f in post if f["ok"])
    ton = max(max(f["tonal_delta"].values()) for f in post)
    alp = max(-np.nanmin(list(f["alpha_delta"].values())) for f in post)
    return {"mount": margin * mount, "coupling": margin * coup, "tonal_db": margin * max(ton, 1.0),
            "alpha": margin * max(alp, 0.05), "margin": margin,
            "healthy_max": {"mount": mount, "coupling": coup, "tonal_db": ton, "alpha": alp}}


def analyze(runs, calib_runs):
    names = runs[0]["names"]
    thr = thresholds([features(r) for r in calib_runs])
    res = {"thresholds": thr, "runs": [], "confusion": {}}
    for r in runs:
        fs = features(r)
        post = [f for f in fs if f["t"] > CAL_END]
        labels = [classify(f, thr, names) for f in post]
        first = next((f["t"] for f, l in zip(post, labels) if l[0] != "healthy"), None)
        last3 = [l[0] for l in labels[-3:]]
        cls = max(set(last3), key=last3.count)
        wh = [l[1] for l in labels[-3:] if l[0] == cls]
        final = (cls, max(set(wh), key=wh.count) if any(wh) else None)
        ff = next((f for f in reversed(post) if f["ok"]), post[-1])
        res["runs"].append({
            "cond": r["cond"], "seed": r["seed"], "sigma": r["sigma"], "final": final[0], "where": final[1],
            "first_flag_t": first, "false_windows": int(sum(l[0] != TRUE_CLASS[r["cond"]] and l[0] != "healthy"
                                                            for l in labels)),
            "final_mount_rel": ff.get("mount_rel"), "final_coupling_rel": ff.get("coupling_rel"),
            "final_graph_gap": ff.get("graph_gap"),
            "graph_gap_cal": float(np.median([f["graph_gap_any"] for f in fs if f["t"] <= CAL_END and "graph_gap_any" in f])),
            "final_stability_gap": ff.get("stability_gap"),
            "stability_gap_cal": float(np.median([f["stability_gap_any"] for f in fs if f["t"] <= CAL_END and "stability_gap_any" in f])),
            "extra_modes_final": ff.get("extra_modes_hz"),
            "final_tonal_delta": ff["tonal_delta"], "final_alpha_delta": ff["alpha_delta"],
            "fiedler_cut": ff.get("fiedler_cut"),
            "series": [{"t": f["t"], "label": l[0], "where": l[1], **({"mount_rel": f["mount_rel"],
                        "coupling_rel": f["coupling_rel"], "graph_gap": f["graph_gap"],
                        "stability_gap": f["stability_gap"]} if f["ok"] else {}),
                        "tonal_delta": f["tonal_delta"], "alpha_delta": f["alpha_delta"]} for f, l in zip(post, labels)],
        })
        key = f"{r['cond']}->{final[0]}"
        res["confusion"][key] = res["confusion"].get(key, 0) + 1
    ok_cls = sum(rr["final"] == TRUE_CLASS[rr["cond"]] for rr in res["runs"])
    ok_loc = [rr["where"] == TRUE_WHERE[rr["cond"]] for rr in res["runs"] if rr["cond"] in TRUE_WHERE]
    res["accuracy"] = {"class": ok_cls / len(res["runs"]), "n": len(res["runs"]),
                       "localisation_structural": float(np.mean(ok_loc)) if ok_loc else None}
    # identification accuracy vs ground truth: mount k_g/m identified in every healthy calibration-seed window
    errs = [e for r in calib_runs for row in r["rows"] for e in row.get("mount_err", [])]
    res["identification"] = {"mount_rel_err_median_abs": float(np.median(np.abs(errs))) if errs else None,
                             "mount_rel_err_p95_abs": float(np.percentile(np.abs(errs), 95)) if errs else None,
                             "values": int(len(errs)),
                             "ok_fraction": float(np.mean([bool(row["watch"].get("ok")) for r in runs + calib_runs
                                                           for row in r["rows"]]))}
    return res


def real_clips():
    """SSI on the user's fan clips (housing pose x, y, rotation vs the floor): are there stable modes?"""
    import warnings
    from scipy.signal import butter, sosfiltfilt
    from vibrovision.realvideo import housing_motion
    from vibrovision.structure import stable_modes
    out = []
    for p in ("data/real/fan_nocoin.mp4", "data/real/fan_coin.mp4"):
        path = os.path.join(ROOT, p)
        if not os.path.exists(path):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = housing_motion(path)
        fps = h["fps"]
        y = np.vstack([np.array(h["_plot"]["d"]).T, 10 * np.array(h["_plot"]["rot"])[None, :]])
        y = sosfiltfilt(butter(4, 1.5, btype="high", fs=fps, output="sos"), y, axis=1)
        lam, sh, diag = stable_modes(y, fps, 3, [h["tone_hz"], 2 * h["tone_hz"]], base_order=8, n_orders=6, step=2)
        out.append({"clip": os.path.basename(p), "fps": fps, "seconds": y.shape[1] / fps, "tone_hz": h["tone_hz"],
                    "housing_rms_px": y[:2].std(axis=1).tolist(),
                    "stable_modes_hz": (np.abs(lam) / (2 * np.pi)).round(2).tolist(),
                    "zeta": (-lam.real / np.abs(lam)).round(4).tolist(), "n_stable": diag["n_stable"]})
    with open(os.path.join(OUT, "structure_real.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="only the SSI check on data/real/*.mp4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--calib-seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if a.real:
        return real_clips()
    pk = os.path.join(OUT, "structure_runs.pkl")
    t0 = time.time()
    if not a.analyze_only:
        jobs = [(c, 100 + s) for c in CONDITIONS for s in range(a.seeds)]
        jobs += [("healthy", 900 + s) for s in range(a.calib_seeds)]
        with Pool(a.workers) as p:
            all_runs = []
            for r in p.imap_unordered(run_one, jobs):
                all_runs.append(r)
                print(f"{r['cond']:10s} seed {r['seed']} done ({time.time() - t0:.0f} s)", flush=True)
        with open(pk, "wb") as fh:
            pickle.dump(all_runs, fh)
    with open(pk, "rb") as fh:
        all_runs = pickle.load(fh)
    calib = [r for r in all_runs if r["seed"] >= 900]
    runs = sorted([r for r in all_runs if r["seed"] < 900], key=lambda r: (list(CONDITIONS).index(r["cond"]), r["seed"]))
    res = analyze(runs, calib)
    res["_meta"] = {"fps": FPS, "duration_s": DUR, "window_s": WIN, "step_s": STEP, "calibration_end_s": CAL_END,
                    "seeds": a.seeds, "calib_seeds": a.calib_seeds, "wall_s": time.time() - t0}
    with open(os.path.join(OUT, "structure_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({"confusion": res["confusion"], "accuracy": res["accuracy"],
                      "identification": res["identification"], "thresholds": res["thresholds"]}, indent=1, default=float))
    for rr in res["runs"]:
        print(f"{rr['cond']:10s} {rr['seed']} -> {rr['final']:10s} {str(rr['where']):32s} first {rr['first_flag_t']} "
              f"false {rr['false_windows']} graph gap {rr['graph_gap_cal']:.2f}->{_f(rr['final_graph_gap'])} "
              f"stability gap {rr['stability_gap_cal']:.2f}->{_f(rr['final_stability_gap'])} extra {rr['extra_modes_final']}")


def _f(x):
    return "n/a" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    main()
