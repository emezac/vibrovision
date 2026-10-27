#!/usr/bin/env python3
"""Reproducible end-to-end evaluation (video rendering -> OpenCV tracking -> indicators/agent).

Design/tuning used development seeds 100-149 (fast truth+noise simulations);
everything here uses *held-out* seeds starting at 1000.

    python eval/run_eval.py               # full (~30-40 min on 2 vCPU)
    python eval/run_eval.py --quick       # smoke version for the notebook

Outputs: results/eval_summary.json, results/eval_windows.pkl
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

import cv2  # noqa: E402

from vibrovision.agent import AgentConfig, HumanGate  # noqa: E402
from vibrovision.detect import SD_FLOOR, alarm_time_for_h, cusum_trace  # noqa: E402
from vibrovision.monitor import REF  # noqa: E402
from vibrovision.pipeline import build_scenario, run_synthetic_agent, sliding_indicators, track_scene  # noqa: E402
from vibrovision.simulate import default_scene, generate_displacements, render_frames  # noqa: E402
from vibrovision.spectral import DIRECTION, INDICATORS  # noqa: E402
from vibrovision.tracking import track_sequence  # noqa: E402

WINDOW_S, STEP_S, CALIB_END = 12.0, 1.0, 60.0
FAULTY = "Rodamiento_B"


# ----------------------------------------------------------------------------- workers
def _init():
    cv2.setNumThreads(1)


def job_indicators(args):
    scenario, seed, extra = args
    t0 = time.time()
    sc = build_scenario(scenario, seed, **extra)
    t, truth, shake, disp, energy = track_scene(sc)
    out = {"scenario": scenario, "seed": seed, "extra": extra}
    ts, ind = sliding_indicators(disp, sc.fps, WINDOW_S, STEP_S, energies=energy)
    out["times"], out["ind"] = ts, ind
    if sc.shake_rms_px > 0:
        _, out["ind_comp"] = sliding_indicators(disp, sc.fps, WINDOW_S, STEP_S, compensate=True, energies=energy)
    # truth alpha for bias analysis (no camera, no noise)
    _, out["ind_truth"] = sliding_indicators({k: v for k, v in truth.items()}, sc.fps, WINDOW_S, STEP_S * 6)
    fault = sc.component(FAULTY).fault
    out["onset_s"] = None if fault is None else fault.onset_s
    out["ring_hz"] = None if fault is None else fault.ring_hz
    out["wall_s"] = time.time() - t0
    return out


def job_agent(args):
    scenario, seed = args
    sc = build_scenario(scenario, seed)
    faulty = {c.name for c in sc.components if c.fault is not None}
    # *simulated* reviewer with ground truth: approves real faults, rejects the rest
    gate = HumanGate(policy=lambda p: (p["component"] in faulty, "simulated-reviewer"))
    t0 = time.time()
    _, _, agent = run_synthetic_agent(scenario, seed, gate=gate)
    tools = {}
    for e in agent.trace:
        tools[e["tool"]] = tools.get(e["tool"], 0) + 1
    esc = [(e["t"], e["roi"]) for e in agent.trace if e["tool"] == "request_human_approval"]
    refined = [(e["t"], e["roi"], e["result"].get("nearest_defect")) for e in agent.trace
               if e["tool"] == "envelope_analysis" and e["result"].get("match")]
    return {"scenario": scenario, "seed": seed, "faulty": sorted(faulty), "escalations": esc,
            "bpfo_identified": refined, "tools": tools,
            "relocated": [e["roi"] for e in agent.trace if e["tool"] == "relocate_roi"],
            "compensation": agent.mon.compensate,
            "work_orders": agent.work_orders, "wall_s": time.time() - t0,
            "track_fps": agent.mon.n * len(agent.mon.boxes) / max(agent.mon.track_seconds, 1e-9),
            "trace_tail": [e for e in agent.trace if e["level"] != "debug"][-12:]}


def job_estimators(seed):
    sc = default_scene(seed=seed, duration_s=20.0)
    t, truth, shake = generate_displacements(sc)
    frames = list(render_frames(sc, truth, shake))
    boxes = {c.name: c.roi for c in sc.components}
    res = {}
    for m in ("ic", "lk", "ecc"):
        t0 = time.perf_counter()
        d, q, n = track_sequence(frames, boxes, m)
        el = time.perf_counter() - t0
        errs = []
        for k in boxes:
            tr = truth[k] - truth[k][:, :1]
            errs.append(np.sqrt(np.mean((d[k] - tr) ** 2)))
        res[m] = {"rmse_px": float(np.mean(errs)), "roi_frames_per_s": n * len(boxes) / el}
    return res


# ----------------------------------------------------------------------------- analysis
def detection_curves(runs, key="ind", hs=np.linspace(0.5, 15, 59)):
    """For each indicator: null false-alarm fraction and fault detection/delay vs h."""
    out = {}
    for ind in INDICATORS:
        null_g, fault = [], []
        for r in runs:
            ts = r["times"]
            cal = ts + WINDOW_S / 2 <= CALIB_END
            for roi, d in r[key].items():
                g = cusum_trace(d[ind], cal, DIRECTION[ind], 0.5, SD_FLOOR[ind], STEP_S / WINDOW_S)
                if roi == FAULTY and r["onset_s"] is not None:
                    fault.append((ts, g, r["onset_s"]))
                else:
                    null_g.append(g.max())
        null_g = np.asarray(null_g)
        fa, det, med = [], [], []
        for h in hs:
            fa.append(float((null_g > h).mean()))
            delays = [alarm_time_for_h(ts, g, h) for ts, g, _ in fault]
            dl = [d - on for d, (_, _, on) in zip(delays, fault) if d is not None]
            det.append(len(dl) / max(len(fault), 1))
            med.append(float(np.median(dl)) if dl else None)
        out[ind] = {"h": hs.tolist(), "false_alarm": fa, "detected": det, "median_delay_s": med,
                    "n_null": int(len(null_g)), "n_fault": len(fault)}
    return out


def at_matched_fa(curves, target=0.05):
    """Pick, per indicator, the smallest h with null FA <= target; report detection."""
    rows = {}
    for ind, c in curves.items():
        idx = next((i for i, f in enumerate(c["false_alarm"]) if f <= target), len(c["h"]) - 1)
        rows[ind] = {"h": c["h"][idx], "false_alarm": c["false_alarm"][idx],
                     "detected": c["detected"][idx], "median_delay_s": c["median_delay_s"][idx]}
    return rows


def nuisance_table(runs):
    """Healthy nuisance runs: does any healthy ROI alarm at the matched-FA thresholds?"""
    return runs


def summarize_agent(agent_runs):
    ag = {}
    for r in agent_runs:
        s = ag.setdefault(r["scenario"], {"runs": 0, "correct_escalation": 0, "false_escalation": 0,
                                          "missed": 0, "bpfo_identified": 0, "detect_t": [], "relocations": 0,
                                          "compensation_enabled": 0, "tool_calls": [], "track_fps": []})
        s["runs"] += 1
        escalated = {roi for _, roi in r["escalations"]}
        faulty = set(r["faulty"])
        s["correct_escalation"] += int(bool(faulty) and faulty <= escalated)
        s["false_escalation"] += len(escalated - faulty)
        s["missed"] += len(faulty - escalated)
        s["bpfo_identified"] += int(any(d == "BPFO" for _, roi, d in r["bpfo_identified"] if roi in faulty))
        s["detect_t"] += [t for t, roi in r["escalations"] if roi in faulty]
        s["relocations"] += len(r["relocated"])
        s["compensation_enabled"] += int(r["compensation"])
        s["tool_calls"].append(sum(r["tools"].values()))
        s["track_fps"].append(r["track_fps"])
    for s in ag.values():
        s["median_detect_t"] = float(np.median(s["detect_t"])) if s["detect_t"] else None
        s["mean_tool_calls"] = float(np.mean(s["tool_calls"]))
        s["mean_track_roi_fps"] = float(np.mean(s["track_fps"]))
    examples = {r["scenario"]: r["trace_tail"] for r in agent_runs if r["seed"] % 1000 == 0}
    return ag, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--only-agent", action="store_true", help="re-run only the agent scenarios and merge")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_main = 3 if a.quick else 12
    n_nuis = 2 if a.quick else 4
    n_agent = 1 if a.quick else 4
    T0 = time.time()
    summary = {"config": {"window_s": WINDOW_S, "step_s": STEP_S, "calib_end_s": CALIB_END,
                          "seeds": "held-out, >= 1000", "quick": a.quick}}

    main_jobs = [("healthy", 1000 + i, {}) for i in range(n_main)] + \
                [("fault", 1000 + i, {"ring_hz": [30.0, 41.0][i % 2]}) for i in range(n_main)]
    nuis_jobs = [(s, 2000 + i, {}) for s in ("healthy_zoom", "healthy_shake", "fault_flicker", "fault_lowlight")
                 for i in range(n_nuis)]
    hard_jobs = [("fault", 3000 + i, {"ring_hz": 18.0}) for i in range(n_nuis)]
    agent_jobs = [(s, 4000 + i) for s in ("healthy", "fault", "fault_shake", "healthy_shake",
                                          "fault_bad_roi", "healthy_zoom") for i in range(n_agent)]

    if a.only_agent:
        with open(os.path.join(a.out, "eval_summary.json")) as fh:
            prev = json.load(fh)
        with Pool(a.workers, initializer=_init) as pool:
            agent_runs = pool.map(job_agent, agent_jobs)
        prev["agent"], prev["agent_examples"] = summarize_agent(agent_runs)
        prev["agent_wall_s"] = time.time() - T0
        with open(os.path.join(a.out, "eval_summary.json"), "w") as fh:
            json.dump(prev, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(json.dumps(prev["agent"], indent=1, default=str))
        return

    with Pool(a.workers, initializer=_init) as pool:
        est = pool.map(job_estimators, [500, 501] if not a.quick else [500])
        ind_runs = pool.map(job_indicators, main_jobs + nuis_jobs + hard_jobs)
        agent_runs = pool.map(job_agent, agent_jobs)

    summary["estimators"] = {m: {k: float(np.mean([e[m][k] for e in est])) for k in est[0][m]} for m in est[0]}
    main_runs = ind_runs[:len(main_jobs)]
    nuis_runs = ind_runs[len(main_jobs):len(main_jobs) + len(nuis_jobs)]
    hard_runs = ind_runs[len(main_jobs) + len(nuis_jobs):]
    curves = detection_curves(main_runs)
    summary["detection_curves"] = curves
    summary["matched_fa_5pct"] = at_matched_fa(curves, 0.05)
    thr = {k: v["h"] for k, v in summary["matched_fa_5pct"].items()}

    # nuisance: alarm outcomes per indicator at the matched thresholds
    nuis = {}
    for r in nuis_runs + hard_runs:
        sc = r["scenario"] + ("_ring18" if r in hard_runs else "")
        ts = r["times"]
        cal = ts + WINDOW_S / 2 <= CALIB_END
        for key in ("ind", "ind_comp"):
            if key not in r:
                continue
            tag = sc + ("+comp" if key == "ind_comp" else "")
            for ind in INDICATORS:
                for roi, d in r[key].items():
                    g = cusum_trace(d[ind], cal, DIRECTION[ind], 0.5, SD_FLOOR[ind], STEP_S / WINDOW_S)
                    alarm = bool(g.max() > thr[ind])
                    kind = "fault" if (roi == FAULTY and r["onset_s"] is not None) else "healthy"
                    cell = nuis.setdefault(tag, {}).setdefault(ind, {"healthy_alarms": 0, "healthy_n": 0,
                                                                     "fault_detected": 0, "fault_n": 0})
                    cell[f"{kind}_n"] += 1
                    if alarm:
                        cell["healthy_alarms" if kind == "healthy" else "fault_detected"] += 1
    summary["nuisance_at_matched_thresholds"] = nuis

    # alpha bias: measured vs truth on healthy ROIs during calibration
    bias = []
    for r in main_runs:
        for roi, d in r["ind"].items():
            if roi == FAULTY and r["onset_s"] is not None:
                continue
            a_meas = np.nanmean(d["alpha"][r["times"] < CALIB_END])
            a_true = np.nanmean(r["ind_truth"][roi]["alpha"][:10])
            bias.append(a_meas - a_true)
    summary["alpha_bias_vs_truth"] = {"mean": float(np.mean(bias)), "sd": float(np.std(bias)), "n": len(bias)}

    summary["agent"], summary["agent_examples"] = summarize_agent(agent_runs)
    summary["wall_s"] = time.time() - T0

    with open(os.path.join(a.out, "eval_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(a.out, "eval_windows.pkl"), "wb") as fh:
        pickle.dump({"main": main_runs, "nuisance": nuis_runs, "hard": hard_runs}, fh)
    print(json.dumps({k: summary[k] for k in ("estimators", "matched_fa_5pct", "alpha_bias_vs_truth")}, indent=1))
    print(json.dumps(summary["agent"], indent=1, default=str))
    print(f"done in {summary['wall_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
