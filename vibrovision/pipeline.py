"""End-to-end runners: synthetic scenarios, real video files, and offline indicator tables."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from .agent import AgentConfig, BearingSpec, HumanGate, InspectionAgent
from .monitor import REF, StreamingMonitor
from .simulate import SceneSpec, default_scene, generate_displacements, render_frames
from .spectral import INDICATORS, SpectralConfig, analyze_window, noise_floor_from_reference
from .tracking import Box, texture_score, track_sequence

# Rodamiento_B sits on the slow gearbox output shaft (24.6 Hz / 12 = 2.05 Hz):
# its outer-race defect frequency (BPFO = 7.29 Hz) is observable at camera frame rates.
DEFAULT_ASSETS = {"Rodamiento_B": BearingSpec(shaft_hz=2.05), "Rodamiento_A": BearingSpec(shaft_hz=24.6)}

SCENARIOS = {
    "healthy":        dict(fault=False),
    "fault":          dict(fault=True),
    "fault_shake":    dict(fault=True, shake_rms_px=0.35),
    "healthy_shake":  dict(fault=False, shake_rms_px=0.35),
    "fault_bad_roi":  dict(fault=True, bad_roi="Rodamiento_B"),
    "healthy_zoom":   dict(fault=False, zoom_at_s=110.0, zoom_gain=1.5),
    "fault_flicker":  dict(fault=True, flicker_amp=0.03),
    "fault_lowlight": dict(fault=True, contrast=0.45, cam_noise=3.0),
    # coupled-structure bench (vibrovision/structure.py): faults that change the machine's Jacobian or not
    "struct_healthy":     dict(fault=False, structure={}),
    "struct_loose_mount": dict(fault=False, structure={"faults": (("mount", ("Rodamiento_B",), 0.3),)}),
    "struct_loose_joint": dict(fault=False, structure={"faults": (("joint", ("Caja_Engranajes", "Rodamiento_A"), 0.2),)}),
    "struct_imbalance":   dict(fault=False, structure={"faults": (("imbalance", ("Motor",), 4.0),)}),
}


def build_scenario(name: str, seed: int = 7, **extra) -> SceneSpec:
    kw = dict(SCENARIOS[name])
    kw.update(extra)
    fault = kw.pop("fault")
    bad = kw.pop("bad_roi", None)
    ring = kw.pop("ring_hz", None)
    st = kw.pop("structure", None)
    if st is not None:
        from .structure import StructFault, StructureSpec
        kw["structure"] = StructureSpec(faults=tuple(StructFault(*f) for f in st.get("faults", ())))
    sc = default_scene(fault=fault, seed=seed, **kw)
    if ring is not None:
        for c in sc.components:
            if c.fault is not None:
                c.fault = replace(c.fault, ring_hz=ring)
    if bad:
        c = sc.component(bad)
        x, y, w, h = c.region
        c.roi = (x + 2, y + h - c.roi[3] - 2, c.roi[2], c.roi[3])   # mostly smooth casting
    return sc


def run_agent_session(frames: Iterable[np.ndarray], fps: float, rois: Dict[str, Box], regions: Dict[str, Box],
                      reference_box: Optional[Box], spec: Optional[SpectralConfig] = None,
                      cfg: Optional[AgentConfig] = None, gate: Optional[HumanGate] = None,
                      assets=None, on_step: Optional[Callable] = None, max_frames: Optional[int] = None):
    spec = spec or SpectralConfig.for_fps(fps)
    cfg = cfg or AgentConfig()
    mon = StreamingMonitor(fps, rois, reference_box, method="ic", buffer_s=cfg.max_window_s + 2.0)
    agent = InspectionAgent(mon, regions, spec, cfg, gate=gate, assets=assets)
    t0 = time.perf_counter()
    for i, fr in enumerate(frames):
        mon.push(fr)
        if agent.due() and mon.n > 1:
            agent.step()
            if on_step:
                on_step(agent)
        if max_frames and i + 1 >= max_frames:
            break
    agent.wall_s = time.perf_counter() - t0
    return agent


def run_synthetic_agent(scenario: str = "fault", seed: int = 7, cfg: Optional[AgentConfig] = None,
                        gate: Optional[HumanGate] = None, **extra):
    cfg = replace(cfg or AgentConfig(), common_scale=True)
    sc = build_scenario(scenario, seed, **extra)
    t, truth, shake = generate_displacements(sc)
    rois = {c.name: c.roi for c in sc.components}
    regions = {c.name: c.region for c in sc.components}
    agent = run_agent_session(render_frames(sc, truth, shake), sc.fps, rois, regions, sc.reference_roi,
                              cfg=cfg, gate=gate, assets=DEFAULT_ASSETS)
    return sc, truth, agent


# ----------------------------------------------------------------------------- offline tables
def sliding_indicators(disp: Dict[str, np.ndarray], fps: float, window_s: float = 12.0, step_s: float = 1.0,
                       spec: Optional[SpectralConfig] = None, compensate: bool = False,
                       energies: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """Per-window indicators for every ROI (noise floor from the static reference)."""
    spec = spec or SpectralConfig.for_fps(fps)
    w, s = int(window_s * fps), int(step_s * fps)
    n = next(iter(disp.values())).shape[1]
    times, out = [], {k: {i: [] for i in INDICATORS + ("ci_width",)} for k in disp if k != REF}
    for i in range(0, n - w + 1, s):
        times.append((i + w / 2) / fps)
        nref = noise_floor_from_reference(disp[REF][:, i:i + w], spec) if REF in disp else 0.0
        for k in out:
            xy = disp[k][:, i:i + w]
            scale = 1.0
            if energies:
                scale = energies[REF] / energies[k]
            nf = nref * scale
            if compensate and REF in disp:
                xy = xy - disp[REF][:, i:i + w]
                nf += nref
            r = analyze_window(xy, spec, nf)
            for ind in out[k]:
                out[k][ind].append(r[ind])
    return np.asarray(times), {k: {i: np.asarray(v) for i, v in d.items()} for k, d in out.items()}


def track_scene(sc: SceneSpec, method: str = "ic"):
    """Render a scenario and track every ROI + the static target (no agent)."""
    t, truth, shake = generate_displacements(sc)
    boxes = {c.name: c.roi for c in sc.components}
    boxes[REF] = sc.reference_roi
    frames = render_frames(sc, truth, shake)
    first = next(render_frames(sc, truth, shake, 0, 1))
    energies = {k: texture_score(first, b)["grad_energy"] * b[2] * b[3] for k, b in boxes.items()}
    disp, qual, n = track_sequence(frames, boxes, method)
    return t, truth, shake, disp, energies


# ----------------------------------------------------------------------------- real video
def read_video(path: str, max_seconds: Optional[float] = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_max = int(max_seconds * fps) if max_seconds else None

    def gen():
        k = 0
        try:
            while n_max is None or k < n_max:
                ok, fr = cap.read()
                if not ok:
                    break
                k += 1
                yield cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        finally:
            cap.release()
    return fps, gen()


def auto_regions(first_gray: np.ndarray, n: int = 4, size: int = 48, exclude: Optional[Box] = None) -> Dict[str, Box]:
    """Pick the n most trackable, non-overlapping square ROIs (Shi-Tomasi)."""
    eig = cv2.cornerMinEigenVal(first_gray.astype(np.float32), blockSize=5, ksize=3)
    integ = cv2.integral(eig)
    H, W = first_gray.shape
    cand = []
    for y in range(4, H - size - 4, 6):
        for x in range(4, W - size - 4, 6):
            s = integ[y + size, x + size] - integ[y, x + size] - integ[y + size, x] + integ[y, x]
            cand.append((s, x, y))
    cand.sort(reverse=True)
    chosen = []
    for s, x, y in cand:
        if all(abs(x - cx) >= size or abs(y - cy) >= size for _, cx, cy in chosen):
            if exclude and not (x + size < exclude[0] or x > exclude[0] + exclude[2] or
                                y + size < exclude[1] or y > exclude[1] + exclude[3]):
                continue
            chosen.append((s, x, y))
        if len(chosen) == n:
            break
    return {f"ROI_{i + 1}": (x, y, size, size) for i, (_, x, y) in enumerate(chosen)}


def probe_reference(frames, fps: float, candidates: Dict[str, Box], seconds: float = 3.0) -> Tuple[str, Dict[str, float]]:
    """Pick the static reference target among textured candidates: the one whose
    measured motion over the first seconds is smallest."""
    k = int(seconds * fps)
    disp, _, _ = track_sequence(frames[:k], candidates, "ic")
    rms = {n: float(np.sqrt(np.sum(d.var(axis=1)))) for n, d in disp.items()}
    return min(rms, key=rms.get), rms


def expand_box(box: Box, shape, margin: int = 16) -> Box:
    """Coarse part region around a ROI (lets the agent search for a better patch)."""
    x, y, w, h = box
    H, W = shape[:2]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    return (x0, y0, min(W - x0, w + 2 * margin), min(H - y0, h + 2 * margin))


def pick_reference(frames, fps: float, rois: Dict[str, Box], n_candidates: int = 16, size: int = 56,
                   seconds: float = 3.0) -> Tuple[Box, Dict[str, float]]:
    """Static reference = the least-moving textured patch *outside* the measured parts.
    Choosing among the parts themselves confuses 'the machine moves' with 'the camera moves'."""
    def overlaps(a, b):
        return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])
    cands = {k: b for k, b in auto_regions(frames[0], n=n_candidates, size=size).items()
             if not any(overlaps(b, r) for r in rois.values())}
    if not cands:
        cands = auto_regions(frames[0], n=n_candidates, size=size)
    name, rms = probe_reference(frames, fps, cands, seconds=seconds)
    return cands[name], rms
