#!/usr/bin/env python3
"""End-to-end check of the *production* agent profile on 30-minute videos with
speed/load variation: one healthy machine and one developing fault per seed.

    python eval/run_production_agent.py [--seeds 2]   (~10 min per video-pair on 1 vCPU)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vibrovision.agent import AgentConfig, HumanGate  # noqa: E402
from vibrovision.pipeline import DEFAULT_ASSETS, run_agent_session  # noqa: E402
from vibrovision.simulate import FaultSpec, default_scene, generate_displacements, render_frames  # noqa: E402


def job(args):
    kind, seed = args
    import cv2
    cv2.setNumThreads(1)
    sc = default_scene(fault=(kind == "fault"), seed=seed, duration_s=1800.0, speed_var=0.03, load_var=0.25)
    if kind == "fault":
        sc.component("Rodamiento_B").fault = FaultSpec(onset_s=720.0, full_s=1800.0, ring_hz=41.0)
    t, truth, shake = generate_displacements(sc)
    faulty = {c.name for c in sc.components if c.fault is not None}
    gate = HumanGate(policy=lambda p: (p["component"] in faulty, "simulated-reviewer"))
    t0 = time.time()
    agent = run_agent_session(render_frames(sc, truth, shake), sc.fps, {c.name: c.roi for c in sc.components},
                              {c.name: c.region for c in sc.components}, sc.reference_roi,
                              cfg=AgentConfig.production(), gate=gate, assets=DEFAULT_ASSETS)
    esc = [(e["t"], e["roi"]) for e in agent.trace if e["tool"] == "request_human_approval"]
    return {"kind": kind, "seed": seed, "escalations": esc, "faulty": sorted(faulty),
            "watch_events": sum(e["tool"] == "watch" for e in agent.trace),
            "xval_rejections": sum(e["tool"] == "cross_validate" and "disagree" in e["decision"] for e in agent.trace),
            "bpfo": [e["t"] for e in agent.trace if e["tool"] == "envelope_analysis" and e["result"].get("match")],
            "wall_s": time.time() - t0,
            "decisions": [{k: e[k] for k in ("t", "tool", "roi", "decision")} for e in agent.trace if e["level"] != "debug"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    jobs = [(k, 8000 + i) for i in range(a.seeds) for k in ("healthy", "fault")]
    with Pool(min(len(jobs), os.cpu_count() or 2)) as pool:
        res = pool.map(job, jobs)
    with open(os.path.join(ROOT, "results", "production_agent.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    for r in res:
        print(r["kind"], r["seed"], "escalations:", r["escalations"], "watch:", r["watch_events"],
              "xval-rejected:", r["xval_rejections"], "bpfo:", r["bpfo"][:1], f"{r['wall_s']:.0f}s")
