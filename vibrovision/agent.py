"""Agentic inspection loop: perception (OpenCV 5) -> decision -> action -> human gate.

Every perception result can change the *next* tool call:

    texture too weak            -> relocate_roi (re-runs OpenCV on buffered frames)
    static target is moving     -> enable_motion_compensation, re-baseline
    alpha CI too wide           -> expand_window (next measurement uses more data)
    CUSUM evidence rising       -> cross_validate with an independent estimator (ECC)
    estimators disagree         -> treat as tracking artefact, relocate instead of alarming
    evidence confirmed          -> envelope_analysis + render_evidence -> request_human_approval
    several parts in view       -> identify_structure (SSI on all ROIs): the machine's Jacobian M^-1 K;
                                   a mount / joint stiffness change opens a localised structural incident
    human approves / rejects    -> create_work_order / re-baseline with higher threshold

The agent never takes an irreversible physical action (stopping a line);
that authority stays with the human reviewer. Every tool call is traced.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .detect import SD_FLOOR, Cusum
from .monitor import REF, StreamingMonitor
from .spectral import (SpectralConfig, analyze_window, defect_line_scores, envelope_spectrum, excess_band, find_tones,
                       psd, tone_power_db)
from .structure import StructuralWatch, force_at
from .spectralmatrix import relative_spectral_evidence
from .theory import estimate_sensor_noise, ic_noise_model
from .tracking import Box, texture_score


# --------------------------------------------------------------------------- asset knowledge
@dataclass
class BearingSpec:
    shaft_hz: float          # shaft speed carrying the bearing
    n_balls: int = 9
    d_over_D: float = 0.21   # ball / pitch diameter
    contact_deg: float = 0.0

    def defect_frequencies(self) -> Dict[str, float]:
        r, c = self.d_over_D, np.cos(np.radians(self.contact_deg))
        fr, n = self.shaft_hz, self.n_balls
        return {"BPFO": n / 2 * fr * (1 - r * c), "BPFI": n / 2 * fr * (1 + r * c),
                "BSF": fr / (2 * r) * (1 - (r * c) ** 2), "FTF": fr / 2 * (1 - r * c), "1X": fr}


@dataclass
class AgentConfig:
    window_s: float = 12.0
    max_window_s: float = 18.0
    step_s: float = 1.0
    calib_s: float = 60.0
    max_ci: float = 0.55          # expand window if the alpha 95 % CI is wider than this
    shrink_ci: float = 0.30       # shrink back when the CI is comfortably narrow
    k: float = 0.5
    # CUSUM increments are weighted by step/window (overlapping windows), so h is in
    # units of independent windows. h_alarm = 6 gave ~1 % false alarms per healthy ROI
    # over 2 min of monitoring on development seeds (see eval/README).
    h_watch: float = 2.0
    h_suspect: float = 4.0
    h_alarm: float = 6.0
    shake_rms_px: float = 0.04    # static target moving more than this => camera shake
    max_camera_motion_px: float = 1.0   # beyond this the camera is not fixed: translation compensation is invalid
    min_tracker_quality: float = 0.3    # below this on most ROIs, motion exceeds what sub-pixel trackers handle
    recommended_fps: float = 120.0
    min_texture_rel: float = 0.6  # relocate if ROI texture (median min-eig) < this x best patch of the part
    min_features_rel: float = 0.35  # ... or if it has < this x the Shi-Tomasi corners of the best patch
    min_quality: float = 0.6
    xval_z: float = 1.0           # ECC alpha must also sit this many baseline sd below the mean
    xval_method: str = "ecc"
    defect_score_db: float = 8.0   # summed line prominence (1x+2x) for a defect-frequency match
    evidence_dir: Optional[str] = None
    magnify_band: Tuple[float, float] = (20.0, 55.0)
    # structural Jacobian watch (vibrovision/structure.py); thresholds from eval/run_structure.py
    structural: bool = False
    struct_window_s: float = 40.0
    struct_every_s: float = 10.0
    struct_calib_n: int = 3
    struct_mount_thr: float = 0.10
    struct_coupling_thr: float = 0.10
    struct_confirm: int = 2
    common_scale: bool = False  # True only for calibrated coplanar ROIs / the synthetic bench
    struct_bootstrap_reps: int = 16
    struct_uncertainty_required: bool = True
    forcing_db: float = 4.0       # rotation-tone power growth (dB) with an unchanged Jacobian -> forcing incident

    @classmethod
    def production(cls, **kw) -> "AgentConfig":
        """Long calibration profile; not a guarantee of an industrial false-alarm rate.

        Historical surrogate runs support h=10 under stationary/load-amplitude changes,
        but not when load changes healthy spectral shape (~12.5 detector alarms/ROI/day).
        This is separate from human escalation and needs field calibration.
        """
        base = dict(calib_s=600.0, step_s=2.0, h_watch=4.0, h_suspect=7.0, h_alarm=10.0)
        base.update(kw)
        return cls(**base)


@dataclass
class RoiState:
    name: str
    region: Box
    phase: str = "CALIBRATING"        # CALIBRATING NOMINAL WATCH SUSPECT AWAITING_HUMAN ESCALATED
    window_s: float = 12.0
    calib: List[Tuple[float, float]] = field(default_factory=list)  # (t, alpha)
    cusum: Optional[Cusum] = None
    h_alarm: float = 6.0
    last: Dict = field(default_factory=dict)
    incident: Optional[str] = None
    baseline_psd: Optional[Tuple[list, list]] = None
    baseline_clip: Optional[Tuple[np.ndarray, np.ndarray]] = None   # (region crops, displacement)
    last_env_t: float = -1e9
    env_match: bool = False


# --------------------------------------------------------------------------- human gate
class HumanGate:
    """Queue of approval requests. `policy` (optional) answers immediately - used
    only for *simulated* reviewers in offline evaluation; in the service the
    decision arrives through the REST API."""

    def __init__(self, policy: Optional[Callable[[dict], Tuple[bool, str]]] = None):
        self.policy = policy
        self.pending: Dict[str, dict] = {}
        self.decisions: Dict[str, dict] = {}

    def request(self, packet: dict) -> None:
        self.pending[packet["incident_id"]] = packet
        if self.policy is not None:
            ok, who = self.policy(packet)
            self.decide(packet["incident_id"], ok, reviewer=who, note="simulated reviewer")

    def decide(self, incident_id: str, approve: bool, reviewer: str = "engineer", note: str = "") -> None:
        if incident_id in self.pending and incident_id not in self.decisions:
            self.decisions[incident_id] = {"approve": bool(approve), "reviewer": reviewer, "note": note,
                                           "decided_at": time.time()}
            self.pending[incident_id]["status"] = "APPROVED" if approve else "REJECTED"
            self.pending[incident_id]["reviewer"] = reviewer

    def poll(self, incident_id: str) -> Optional[dict]:
        return self.decisions.get(incident_id)


# --------------------------------------------------------------------------- the agent
class InspectionAgent:
    def __init__(self, monitor: StreamingMonitor, regions: Dict[str, Box], spec: SpectralConfig,
                 cfg: AgentConfig = AgentConfig(), gate: Optional[HumanGate] = None,
                 assets: Optional[Dict[str, BearingSpec]] = None, planner=None):
        self.mon, self.spec, self.cfg = monitor, spec, cfg
        self.gate = gate or HumanGate()
        self.assets = assets or {}
        self.planner = planner
        self.rois = {n: RoiState(n, regions[n], window_s=cfg.window_s, h_alarm=cfg.h_alarm)
                     for n in regions}
        self.trace: List[dict] = []
        self.work_orders: List[dict] = []
        self.booted = False
        self.next_step_t = cfg.step_s
        self.setup_invalid: Optional[dict] = None
        self.struct: Optional[StructuralWatch] = None
        self.struct_next_t = cfg.struct_window_s
        self.struct_hits = 0
        self.struct_incident: Optional[str] = None
        self.tone_cal: Dict[str, List[float]] = {}
        self.tone_hits: Dict[str, int] = {}
        self.force_incident: Optional[str] = None
        self.noise_model: Dict[str, dict] = {}

    # ------------------------------------------------------------- tracing
    def _log(self, tool: str, roi: Optional[str], args: dict, result: dict, decision: str = "",
             dt: float = 0.0, level: str = "info") -> dict:
        ev = {"t": round(self.mon.t, 2), "tool": tool, "roi": roi, "args": args, "result": result,
              "decision": decision, "latency_ms": round(1e3 * dt, 2), "level": level}
        self.trace.append(ev)
        return ev

    def _call(self, tool: str, roi: Optional[str], fn, args: dict, level: str = "info"):
        t0 = time.perf_counter()
        res = fn(**args)
        return res, time.perf_counter() - t0

    # ------------------------------------------------------------- tools (each runs OpenCV / DSP)
    def tool_assess_roi_quality(self, roi: str) -> dict:
        g = self.mon.buffered(1.0 / self.mon.fps)[-1]
        ts = texture_score(g, self.mon.boxes[roi])
        ts["tracker_quality"] = self.mon.quality(roi, 2.0) if self.mon.disp[roi] else 1.0
        ts["box"] = list(self.mon.boxes[roi])
        return ts

    def tool_relocate_roi(self, roi: str) -> dict:
        old = self.mon.boxes[roi]
        new, score = self.mon.best_box_in(self.rois[roi].region, (old[2], old[3]))
        self.mon.relocate(roi, new)
        return {"old_box": list(old), "new_box": list(new), "texture_score": round(score, 1),
                "backfilled_s": round(self.mon.available_s(roi), 2)}

    def tool_check_camera_motion(self) -> dict:
        r = self.mon.reference_motion(min(4.0, self.mon.t))
        r["shake"] = r["rms_px"] > self.cfg.shake_rms_px
        return r

    def tool_enable_motion_compensation(self) -> dict:
        self.mon.compensate = True
        return {"compensate": True}

    def tool_measure_spectrum(self, roi: str, window_s: float) -> dict:
        xy = self.mon.window(roi, window_s)
        nf = self.mon.noise_floor(roi, window_s, self.spec)
        r = analyze_window(xy, self.spec, nf)
        r["noise_psd"] = nf
        r["tracker_quality"] = self.mon.quality(roi, window_s)
        return r

    def tool_cross_validate(self, roi: str, window_s: float) -> dict:
        xy, _ = self.mon.retrack(roi, window_s, self.cfg.xval_method)
        nf = self.mon.noise_floor(roi, window_s, self.spec)
        r = analyze_window(xy, self.spec, nf)
        return {"method": self.cfg.xval_method, "alpha": r["alpha"], "ci": [r["ci_lo"], r["ci_hi"]]}

    def tool_envelope_analysis(self, roi: str, window_s: float) -> dict:
        """Demodulate the band where the spectrum grew most vs. baseline (perception
        chooses the band), then look for the impact repetition rate."""
        st = self.rois[roi]
        xy = self.mon.window(roi, window_s)
        band = None
        if st.baseline_psd is not None:
            f, P = psd(xy - xy.mean(axis=1, keepdims=True), self.spec)
            fb, Pb = np.asarray(st.baseline_psd[0]), np.asarray(st.baseline_psd[1])
            Pb = np.interp(f, fb, Pb)
            _, band = excess_band(f, P, Pb, self.spec)
        env = envelope_spectrum(xy, self.spec, band=band)
        out = {"impact_rate_hz": env["peak_hz"], "prominence_db": round(env["prominence_db"], 1),
               "demod_band_hz": env.get("band_hz"), "notched_tones_hz": env.get("notched_tones_hz")}
        asset = self.assets.get(roi)
        if asset is not None:
            # hypothesis test against the bearing's kinematic defect frequencies
            cands = {k: v for k, v in asset.defect_frequencies().items() if 2.0 <= v <= 20.0}
            scores = defect_line_scores(env, cands)
            if scores:
                ranked = sorted(scores.items(), key=lambda kv: -kv[1])
                best, sc = ranked[0]
                margin = sc - (ranked[1][1] if len(ranked) > 1 else 0.0)
                ok = sc >= self.cfg.defect_score_db and margin >= 3.0
                out.update(defect_scores_db={k: round(v, 1) for k, v in scores.items()},
                           nearest_defect=best, nearest_hz=round(cands[best], 2), match=bool(ok))
                if ok:
                    out["impact_rate_hz"] = round(cands[best], 2)
        return out

    def _clip(self, roi: str, seconds: float = 2.0):
        from .tracking import crop
        frames = self.mon.buffered(seconds)
        crops = np.stack([crop(f, self.rois[roi].region, 6) for f in frames])
        return crops, self.mon.window(roi, len(frames) / self.mon.fps)

    def tool_render_evidence(self, roi: str, window_s: float, band=None) -> dict:
        out = {}
        d = self.cfg.evidence_dir
        st = self.rois[roi]
        band = tuple(band) if band else (self.cfg.magnify_band if self.spec.fs >= 110 else self.spec.hi_band)
        if d:
            os.makedirs(d, exist_ok=True)
            from .magnify import comparison_panel
            tag = f"{roi}_t{int(self.mon.t)}"
            if st.baseline_clip is not None:
                nc, nd = self._clip(roi)
                bc, bd = st.baseline_clip
                p = os.path.join(d, f"{tag}_xt_magnified.png")
                comparison_panel(bc, bd, nc, nd, self.mon.fps, band, path=p)
                out["xt_slice_png"] = p
            p2 = os.path.join(d, f"{tag}_spectrum.png")
            self._plot_evidence(roi, window_s, band, p2)
            out["spectrum_png"] = p2
        out["magnified_band_hz"] = list(band)
        return out

    def tool_predict_noise_floor(self, roi: str) -> dict:
        """Image-Jacobian theory: tracker noise from one frame + sensor noise (vibrovision/theory.py)."""
        frames = self.mon.buffered(min(2.0, self.mon.t))
        if getattr(self, "_sigma", None) is None:
            # one estimate per session, on the static reference (moving parts outside the ROIs - fan blades,
            # people - violate the small-motion model); whole frame only if there is no reference
            self._sigma = estimate_sensor_noise(list(frames), self.mon.boxes.get(REF))["sigma"]
        sig = self._sigma
        m = ic_noise_model(frames[-1], self.mon.boxes[roi], sig, self.mon.fps)
        # smallest broadband vibration (rms, px) that clears the floor by 10 dB across the fit band
        bw = self.spec.fit_band[1] - self.spec.fit_band[0]
        m["min_detectable_rms_px"] = float(np.sqrt(10.0 * m["floor_psd"] * bw))
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}

    def tool_identify_structure(self, window_s: float) -> dict:
        names = [n for n in self.rois if n != REF]
        disp = {n: self.mon.window(n, window_s) for n in names}
        tones = set()
        for n in names:
            f, P = psd(disp[n], self.spec)
            tones.update(round(t, 2) for t, _ in find_tones(f, P, self.spec, top=3))
        if self.struct is None:
            self.struct = StructuralWatch(names, self.mon.fps, sorted(tones), calib_n=self.cfg.struct_calib_n,
                                          mount_thr=self.cfg.struct_mount_thr,
                                          coupling_thr=self.cfg.struct_coupling_thr,
                                          common_scale=self.cfg.common_scale,
                                          bootstrap_reps=self.cfg.struct_bootstrap_reps,
                                          uncertainty_required=self.cfg.struct_uncertainty_required)
        else:
            self.struct.tone_hz = sorted(set(self.struct.tone_hz) | tones)
        ev = self.struct.update(disp, round(self.mon.t, 2))
        joint = np.concatenate([disp[n] for n in names], axis=0)
        if not hasattr(self, "_joint_spectral_baseline"):
            self._joint_spectral_baseline = joint.copy()
        try:
            ev["joint_spectral"] = relative_spectral_evidence(self._joint_spectral_baseline, joint, self.mon.fps)
        except (ValueError, np.linalg.LinAlgError) as exc:
            ev["joint_spectral"] = {"available": False, "reason": str(exc)}
        # forcing check on the same windows: power of the rotation lines found during calibration
        if ev.get("phase") == "calibrating" or not hasattr(self, "_cal_tones"):
            self._cal_tones = sorted(self.struct.tone_hz)
        if self._cal_tones:
            tail = {n: disp[n][:, -int(12 * self.mon.fps):] for n in names}
            ev["line_db"] = {f"{f0:.2f}": 10 * np.log10(sum(10 ** (tone_power_db(tail[n], self.spec, [f0]) / 10)
                                                            for n in names)) for f0 in self._cal_tones}
            if ev.get("ok") and ev.get("jacobian"):
                src = {}
                for f0 in self._cal_tones:
                    mags = [force_at(disp, names, self.mon.fps, f0, J["MinvK"], J["MinvC"], int(ax))[1]
                            for ax, J in ev["jacobian"].items()]
                    src[f"{f0:.2f}"] = dict(zip(names, np.round(np.mean(mags, 0), 1).tolist()))
                ev["line_source"] = src
        return ev

    def tool_request_human_approval(self, roi: str, evidence: dict) -> dict:
        st = self.rois[roi]
        inc = f"INC-{roi}-{int(self.mon.t * 100):07d}"
        packet = {
            "incident_id": inc, "t_s": round(self.mon.t, 2), "component": roi,
            "roi_box": list(self.mon.boxes[roi]),
            "alpha_now": round(st.last.get("alpha", np.nan), 3),
            "alpha_ci95": [round(st.last.get("ci_lo", np.nan), 3), round(st.last.get("ci_hi", np.nan), 3)],
            "alpha_baseline": round(st.cusum.mu, 3), "alpha_baseline_sd": round(st.cusum.sd, 3),
            "cusum": round(st.cusum.g, 2), "threshold": st.h_alarm,
            "band_ratio_now": round(st.last.get("band_ratio", np.nan), 3),
            "rms_px_now": round(st.last.get("rms", np.nan), 4),
            "evidence": evidence,
            "fault_class": self._fault_class(),
            "recommended_action": self._recommendation(evidence),
            "autonomy": "advisory-only: no machine control without human sign-off",
            "status": "AWAITING_ENGINEER_SIGN_OFF",
        }
        self.gate.request(packet)
        st.incident = inc
        return packet

    def tool_create_work_order(self, roi: str, decision: dict) -> dict:
        wo = {"work_order": f"WO-{len(self.work_orders) + 1:04d}", "incident_id": self.rois[roi].incident,
              "component": roi, "approved_by": decision["reviewer"], "t_s": round(self.mon.t, 2)}
        self.work_orders.append(wo)
        return wo

    # ------------------------------------------------------------- helpers
    def _fault_class(self) -> str:
        """S_yy = H S_ff H^*: a spectral-shape alarm with an unchanged Jacobian is a forcing change."""
        last = next((e for e in reversed(self.struct.history) if e.get("ok")), None) if self.struct else None
        if last is None or last.get("phase") != "watching":
            return "unclassified (structural watch off or not calibrated)"
        if last.get("change"):
            return f"structural ({last['change']['kind']}) + spectral-shape change"
        return "forcing / impacts: spectral shape changed, identified Jacobian M^-1 K unchanged"

    def _recommendation(self, ev: dict) -> str:
        env = ev.get("envelope", {})
        if env.get("match"):
            return (f"Envelope lines at {env['nearest_defect']} ({env['nearest_hz']} Hz) and harmonics: "
                    f"schedule bearing inspection (outer race / lubrication) "
                    f"at next planned stop; confirm with contact accelerometer.")
        return "Broadband spectral-shape change without a matched defect frequency: confirm with a contact sensor."

    def _plot_evidence(self, roi: str, window_s: float, band, path: str) -> None:
        # Figure API (no pyplot): never switches the host's matplotlib backend
        from matplotlib.figure import Figure
        from scipy.signal import butter, sosfiltfilt
        st = self.rois[roi]
        xy = self.mon.window(roi, window_s)
        f, P = psd(xy - xy.mean(axis=1, keepdims=True), self.spec)
        nf = self.mon.noise_floor(roi, window_s, self.spec)
        fig = Figure(figsize=(9.5, 3.4), dpi=110)
        ax, ax2 = fig.subplots(1, 2)
        if st.baseline_psd:
            ax.loglog(st.baseline_psd[0][1:], st.baseline_psd[1][1:], color="#2a7", lw=1.1, label="baseline")
        ax.loglog(f[1:], P[1:], color="#c33", lw=1.1, label="now")
        fb = np.array(self.spec.fit_band)
        a_now, a_base = st.last.get("alpha", np.nan), st.cusum.mu
        fm = np.sqrt(fb[0] * fb[1])
        near = (f >= fm / 1.4) & (f <= fm * 1.4)
        ref = float(np.median(P[near]))
        for a, c, lab in ((a_base, "#2a7", "baseline"), (a_now, "#c33", "now")):
            ax.loglog(fb, ref * (fb / fm) ** (-a), "--", color=c, lw=1.4, label=f"alpha {lab} = {a:.2f}")
        ax.axhline(nf, color="gray", ls=":", label="tracker noise floor")
        ax.axvspan(band[0], band[1], color="orange", alpha=0.12, label="excess band")
        ax.set_xlabel("Hz"); ax.set_ylabel("px$^2$/Hz"); ax.legend(fontsize=6.5); ax.grid(alpha=.3, which="both")
        ax.set_title(f"{roi}: spectrum (window {window_s:.0f}s)", fontsize=9)
        sos = butter(4, [band[0], min(band[1], 0.45 * self.mon.fps)], btype="bandpass", fs=self.mon.fps, output="sos")
        n = int(2 * self.mon.fps)
        tt = np.arange(n) / self.mon.fps
        if st.baseline_clip is not None:
            ax2.plot(tt, sosfiltfilt(sos, st.baseline_clip[1][0])[-n:], color="#2a7", lw=0.8, label="baseline")
        ax2.plot(tt, sosfiltfilt(sos, self.mon.window(roi, 4.0)[0])[-n:], color="#c33", lw=0.8, label="now")
        ax2.set_xlabel("s"); ax2.set_ylabel("px (band-passed)"); ax2.legend(fontsize=7); ax2.grid(alpha=.3)
        ax2.set_title(f"x-displacement {band[0]:.0f}-{band[1]:.0f} Hz: impact bursts", fontsize=9)
        fig.tight_layout(); fig.savefig(path)

    def _recalibrate(self, st: RoiState, seconds_back: float) -> None:
        """Recompute baseline alphas from stored history with the current settings
        (e.g. after motion compensation was switched on)."""
        fps = self.mon.fps
        w = int(st.window_s * fps)
        nf = self.mon.noise_floor(st.name, st.window_s, self.spec)
        alphas, end = [], self.mon.n
        first = self.mon.start[st.name]
        while end - w >= first and (self.mon.n - end) / fps < seconds_back:
            xy = self.mon.window(st.name, st.window_s, end=end)
            alphas.append(analyze_window(xy, self.spec, nf)["alpha"])
            end -= int(self.cfg.step_s * fps)
        if alphas:
            st.cusum = Cusum(direction="down", k=self.cfg.k, h=st.h_alarm, sd_floor=SD_FLOOR["alpha"],
                             weight=self.cfg.step_s / st.window_s)
            st.cusum.calibrate(alphas)

    # ------------------------------------------------------------- main loop
    def due(self) -> bool:
        return self.mon.t + 1e-9 >= self.next_step_t

    def step(self) -> None:
        self.next_step_t += self.cfg.step_s
        if self.setup_invalid:
            return
        if not self.booted:
            self._bootstrap()
        if self._validate_setup():
            return
        self._check_camera()
        for st in self.rois.values():
            self._step_roi(st)
        if self.cfg.structural and len(self.rois) >= 2 and self.mon.t + 1e-9 >= self.struct_next_t:
            self.struct_next_t += self.cfg.struct_every_s
            self._step_structure()
        self._poll_machine_incidents()

    def _poll_machine_incidents(self) -> None:
        """Reviewer decisions on machine-level (structural / forcing) incidents."""
        for attr in ("struct_incident", "force_incident"):
            inc = getattr(self, attr)
            if inc is None or inc in getattr(self, "_handled", set()):
                continue
            dec = self.gate.poll(inc)
            if dec is None:
                continue
            self._handled = getattr(self, "_handled", set()) | {inc}
            pkt = self.gate.pending[inc]
            if dec["approve"]:
                wo = {"work_order": f"WO-{len(self.work_orders) + 1:04d}", "incident_id": inc,
                      "component": pkt["component"], "fault_class": pkt.get("fault_class"),
                      "approved_by": dec["reviewer"], "t_s": round(self.mon.t, 2)}
                self.work_orders.append(wo)
                self._log("create_work_order", None, {}, wo, f"approved by {dec['reviewer']}")
            else:
                if attr == "struct_incident":
                    self.cfg.struct_mount_thr *= 1.5
                    self.cfg.struct_coupling_thr *= 1.5
                    if self.struct:
                        self.struct.mount_thr, self.struct.coupling_thr = self.cfg.struct_mount_thr, self.cfg.struct_coupling_thr
                    self.struct_hits = 0
                else:
                    self.cfg.forcing_db += 2.0
                    self.tone_hits = {}
                setattr(self, attr, None)
                self._log("human_rejected", None, dec, {"incident": inc},
                          "reviewer rejected a machine-level incident: its threshold raised")

    def _step_structure(self) -> None:
        ev, dt = self._call("identify_structure", None, self.tool_identify_structure,
                            {"window_s": self.cfg.struct_window_s})
        if not ev["ok"]:
            self._log("identify_structure", None, {"window_s": self.cfg.struct_window_s}, ev,
                      "No structural diagnosis: " + ev.get("reason", "insufficient measurement support"), dt)
            return
        if ev["phase"] == "calibrating":
            for n, v in ev.get("line_db", {}).items():
                self.tone_cal.setdefault(n, []).append(v)
            self._log("identify_structure", None, {"window_s": self.cfg.struct_window_s}, ev,
                      "baseline of the machine Jacobian M^-1 K", dt, level="debug")
            return
        if ev.get("decision") in ("abstain", "collect_more_data", "change_unlocalized"):
            self.struct_hits = 0
            self._log("structural_abstain", None, {}, ev,
                      "Evidence is not precise enough to localize a structural change; collect more stationary footage", dt)
            return
        self._check_forcing(ev)
        ch = ev.get("change")
        location_key = (ch["kind"], ch.get("part") or ch.get("between")) if ch else None
        previous_key = getattr(self, "_struct_candidate_key", None)
        self.struct_hits = (self.struct_hits + 1 if location_key == previous_key else 1) if ch else 0
        self._struct_candidate_key = location_key
        if not ch:
            self._log("identify_structure", None, {}, ev, "Jacobian unchanged (mounts and couplings in range)",
                      dt, level="debug")
            return
        where = ch.get("part") or ch.get("between")
        self._log("identify_structure", None, {}, ev,
                  f"{ch['kind']} stiffness change {ch['change']:+.0%} at {where} "
                  f"({self.struct_hits}/{self.cfg.struct_confirm})", dt)
        if self.struct_hits >= self.cfg.struct_confirm and self.struct_incident is None:
            self.struct_incident = f"INC-STRUCT-{int(self.mon.t * 100):07d}"
            if ch["kind"] in ("mount", "local_stiffness"):
                action = (f"Check the mounting of {where} (bolts, soft foot, isolator): its identified mount "
                          f"stiffness k/m changed {ch['change']:+.0%} while the other parts did not." if ch["kind"] == "mount" else
                          f"Inspect {where}: local dynamic stiffness changed {ch['change']:+.0%}. "
                          "Unknown relative camera scales prevent an absolute mounting-stiffness diagnosis.")
            else:
                a, b = where.split("|")
                action = (f"Check the joint / coupling between {a} and {b}: identified coupling changed "
                          f"{ch['change']:+.0%}. The location comes from a resolved operator entry; "
                          "the graph partition is supporting descriptive evidence.")
            pkt = {"incident_id": self.struct_incident, "t_s": round(self.mon.t, 2), "component": where,
                   "fault_class": f"structural ({ch['kind']})", "evidence": {"structure": ev},
                   "recommended_action": action,
                   "autonomy": "advisory-only: no machine control without human sign-off",
                   "status": "AWAITING_ENGINEER_SIGN_OFF"}
            self.gate.request(pkt)
            self._log("request_human_approval", None, {"incident": self.struct_incident}, {"component": where},
                      "structural incident opened (localised from the identified Jacobian)")

    def _check_forcing(self, ev: dict) -> None:
        """Taxonomy, forcing branch: a rotation line grows while the identified Jacobian does not change.
        The source is located by inverting the identified dynamics at that line (structure.force_at)."""
        if not self.tone_cal or ev.get("change") or ev.get("candidate_change") or "line_db" not in ev:
            return
        delta = {f: v - float(np.median(self.tone_cal[f])) for f, v in ev["line_db"].items() if f in self.tone_cal}
        for f, d in delta.items():
            self.tone_hits[f] = self.tone_hits.get(f, 0) + 1 if d > self.cfg.forcing_db else 0
        line = max(delta, key=delta.get)
        if self.tone_hits.get(line, 0) >= self.cfg.struct_confirm and self.force_incident is None:
            src = ev.get("line_source", {}).get(line)
            where = max(src, key=src.get) if src and self.cfg.common_scale else None
            self.force_incident = f"INC-FORCE-{int(self.mon.t * 100):07d}"
            pkt = {"incident_id": self.force_incident, "t_s": round(self.mon.t, 2), "component": where or "machine",
                   "fault_class": "tonal forcing candidate (requires review)",
                   "evidence": {"line_hz": float(line), "line_db_change": {k: round(v, 2) for k, v in delta.items()},
                                "force_per_unit_mass_at_line": src,
                                "structure": {k: ev.get(k) for k in ("mount_rel", "coupling_rel", "graph_gap")}},
                   "recommended_action": (f"The {line} Hz rotation line grew {delta[line]:+.1f} dB while the mounts and "
                                          f"operator have no resolved structural change. Check balance / alignment; "
                                          f"candidate source: {where or 'not localized with current scales'}. "
                                          "Changing excitation can also explain spectral changes."),
                   "autonomy": "advisory-only: no machine control without human sign-off",
                   "status": "AWAITING_ENGINEER_SIGN_OFF"}
            self.gate.request(pkt)
            self._log("request_human_approval", where, {"incident": self.force_incident}, pkt["evidence"],
                      f"forcing incident: {line} Hz line {delta[line]:+.1f} dB, Jacobian unchanged, force on {where}")

    def _validate_setup(self) -> bool:
        """Refuse to diagnose when the footage cannot support sub-pixel measurement.

        Two different situations are separated:
          * the static reference moves -> the camera is not fixed -> setup invalid, ask to re-record;
          * the reference is still but the parts lose tracking -> the *machine* moves by whole
            pixels (loose mounting, a free-standing machine walking) -> that is itself an
            anomaly: open an incident for a human instead of blaming the camera.
        Returns True when normal monitoring must stop."""
        if REF not in self.mon.boxes:
            return False
        win = min(4.0, self.mon.t)
        r = self.mon.reference_motion(win)
        quals = {n: self.mon.quality(n, win) for n in self.rois}
        q = float(np.median(list(quals.values())))
        camera_moves = r["rms_px"] > self.cfg.max_camera_motion_px
        lost = [n for n, v in quals.items() if v < self.cfg.min_tracker_quality]
        if not camera_moves and not lost:
            return False
        if camera_moves:
            self.setup_invalid = {
                "t_s": round(self.mon.t, 2), "kind": "camera_not_fixed", "reference_rms_px": r["rms_px"],
                "median_tracker_quality": q,
                "problems": [f"static reference moves {r['rms_px']:.1f} px RMS (limit {self.cfg.max_camera_motion_px} px): camera is not fixed"],
                "request": "Re-record with the camera on a tripod or rigid support that does not touch the machine's "
                           "table, exposure/focus locked, >= 120 fps, >= 60 s per condition, original file "
                           "(not re-compressed by a messaging app)."}
            for st in self.rois.values():
                st.phase = "INVALID_SETUP"
            self._log("validate_setup", None, {}, self.setup_invalid,
                      "camera moves -> measurement invalid -> no diagnosis; ask operator to re-record", 0.0)
            return True
        if len(lost) * 2 < len(self.rois):         # isolated tracking loss: handled per ROI (relocation)
            return False
        # camera still, parts lost: gross motion of the machine
        self.setup_invalid = {
            "t_s": round(self.mon.t, 2), "kind": "gross_part_motion", "reference_rms_px": r["rms_px"],
            "median_tracker_quality": q, "parts_lost": lost,
            "problems": [f"static reference steady ({r['rms_px']:.2f} px RMS) but {len(lost)}/{len(self.rois)} parts "
                         f"lost sub-pixel tracking (median quality {q:.2f}): the machine itself moves by whole pixels"],
            "request": "Inspect mounting now: a part moving by whole pixels while the camera is still indicates "
                       "loose fixing, a free-standing machine walking, or a severe imbalance."}
        inc = f"INC-GROSS-{int(self.mon.t * 100):07d}"
        packet = {"incident_id": inc, "t_s": round(self.mon.t, 2), "component": ",".join(lost),
                  "kind": "gross_part_motion", "evidence": {"reference_rms_px": r["rms_px"], "tracker_quality": quals},
                  "recommended_action": self.setup_invalid["request"],
                  "autonomy": "advisory-only: no machine control without human sign-off",
                  "status": "AWAITING_ENGINEER_SIGN_OFF"}
        self.gate.request(packet)
        for n, st in self.rois.items():
            st.phase = "AWAITING_HUMAN" if n in lost else "INVALID_SETUP"
            if n in lost:
                st.incident = inc
        self._log("validate_setup", None, {}, self.setup_invalid,
                  "camera steady but parts move beyond tracking range -> gross anomaly -> human sign-off", 0.0)
        self._log("request_human_approval", ",".join(lost), {}, {"incident_id": inc,
                  "recommended_action": packet["recommended_action"]}, "human sign-off required", 0.0)
        return True

    def _bootstrap(self) -> None:
        self.booted = True
        if self.mon.fps < self.cfg.recommended_fps:
            self._log("validate_setup", None, {"fps": round(self.mon.fps, 2)},
                      {"observable_band_hz": [0.0, round(self.mon.fps / 2, 1)]},
                      f"low frame rate: only < {self.mon.fps / 2:.0f} Hz observable; higher frequencies alias")
        for name in self.rois:
            try:
                nm, dt = self._call("predict_noise_floor", name, self.tool_predict_noise_floor, {"roi": name})
            except Exception as e:                 # theory is advisory; never block the loop
                nm, dt = {"error": str(e)}, 0.0
            self.noise_model[name] = nm
            if "floor_psd" in nm:
                self._log("predict_noise_floor", name, {}, nm,
                          f"Cramer-Rao: >= {nm['crb_rms_px']:.4f} px/frame; tracker {nm['ic_rms_px']:.4f} px/frame "
                          f"(sensor sigma {nm['sigma']:.2f}); smallest detectable broadband vibration "
                          f"~{nm['min_detectable_rms_px']:.3f} px rms (white-noise model: a lower bound for compressed video)", dt)
        for name, st in self.rois.items():
            q, dt = self._call("assess_roi_quality", name, self.tool_assess_roi_quality, {"roi": name})
            box = self.mon.boxes[name]
            best, best_score = self.mon.best_box_in(st.region, (box[2], box[3]))
            g = self.mon.buffered(1.0 / self.mon.fps)[-1]
            tb = texture_score(g, best)
            rel = q["min_eig_p50"] / max(tb["min_eig_p50"], 1e-9)
            relf = q["n_features"] / max(tb["n_features"], 1)
            q["texture_vs_best"], q["features_vs_best"] = round(rel, 2), round(relf, 2)
            low = rel < self.cfg.min_texture_rel or relf < self.cfg.min_features_rel
            self._log("assess_roi_quality", name, {}, q,
                      f"texture {rel:.0%} / corners {relf:.0%} of best patch on this part -> relocate" if low else "ok", dt)
            if low:
                r, dt = self._call("relocate_roi", name, self.tool_relocate_roi, {"roi": name})
                self._log("relocate_roi", name, {"region": list(st.region)}, r,
                          "ROI moved to the most trackable patch of the same part; history back-filled", dt)
                q2 = self.tool_assess_roi_quality(name)
                self._log("assess_roi_quality", name, {}, q2, "re-check after relocation")

    def _check_camera(self) -> None:
        if REF not in self.mon.boxes or self.mon.compensate:
            return
        r, dt = self._call("check_camera_motion", None, self.tool_check_camera_motion, {})
        if r["shake"]:
            self._log("check_camera_motion", None, {}, r, "static target moves -> camera shake", dt)
            res, dt = self._call("enable_motion_compensation", None, self.tool_enable_motion_compensation, {})
            self._log("enable_motion_compensation", None, {}, res,
                      "subtract reference-target motion from every ROI; re-baseline", dt)
            for st in self.rois.values():
                if st.cusum is not None:
                    self._recalibrate(st, self.cfg.calib_s)
                    self._log("recalibrate", st.name, {}, {"mu": st.cusum.mu, "sd": st.cusum.sd},
                              "baseline recomputed from stored history with compensation")
                st.calib = []  # calibration still in progress restarts with compensated data

    def _step_roi(self, st: RoiState) -> None:
        cfg = self.cfg
        if self.mon.available_s(st.name) < st.window_s:
            return
        if st.phase in ("ESCALATED", "AWAITING_HUMAN") and not st.env_match \
                and self.mon.t - st.last_env_t >= 10.0:
            self._refine_diagnosis(st)
        if st.phase == "ESCALATED":
            return
        m, dt = self._call("measure_spectrum", st.name, self.tool_measure_spectrum,
                           {"roi": st.name, "window_s": st.window_s})
        st.last = m
        self._log("measure_spectrum", st.name, {"window_s": st.window_s},
                  {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                   if k in ("alpha", "ci_lo", "ci_hi", "band_ratio", "rms", "snr_db", "tracker_quality")},
                  "", dt, level="debug")

        # perception quality gates --------------------------------------------------
        if m["tracker_quality"] < cfg.min_quality:
            r, dt = self._call("relocate_roi", st.name, self.tool_relocate_roi, {"roi": st.name})
            self._log("relocate_roi", st.name, {}, r, f"tracker quality {m['tracker_quality']:.2f} < {cfg.min_quality}", dt)
            return
        ciw = m.get("ci_width", np.nan)
        if np.isfinite(ciw) and ciw > cfg.max_ci and st.window_s < cfg.max_window_s:
            new = min(cfg.max_window_s, st.window_s * 1.5)
            self._log("expand_window", st.name, {"from_s": st.window_s, "to_s": new},
                      {"ci_width": round(ciw, 3)}, "alpha uncertainty too high -> integrate longer")
            st.window_s = new
        elif np.isfinite(ciw) and ciw < cfg.shrink_ci and st.window_s > cfg.window_s:
            new = max(cfg.window_s, st.window_s / 1.5)
            self._log("shrink_window", st.name, {"from_s": st.window_s, "to_s": new},
                      {"ci_width": round(ciw, 3)}, "uncertainty low -> restore responsiveness")
            st.window_s = new

        # calibration ----------------------------------------------------------------
        if st.phase == "CALIBRATING":
            if np.isfinite(m["alpha"]):
                st.calib.append((self.mon.t, m["alpha"]))
            if self.mon.t >= cfg.calib_s and len(st.calib) >= 5:
                st.cusum = Cusum(direction="down", k=cfg.k, h=st.h_alarm, sd_floor=SD_FLOOR["alpha"],
                                 weight=cfg.step_s / st.window_s)
                st.cusum.calibrate([a for _, a in st.calib])
                xy = self.mon.window(st.name, st.window_s)
                f, P = psd(xy - xy.mean(axis=1, keepdims=True), self.spec)
                st.baseline_psd = (f.tolist(), P.tolist())
                st.baseline_clip = self._clip(st.name, 4.0)
                st.phase = "NOMINAL"
                self._log("calibrate", st.name, {"n_windows": len(st.calib)},
                          {"alpha_mu": round(st.cusum.mu, 3), "alpha_sd": round(st.cusum.sd, 3)},
                          "baseline learned; monitoring starts")
            return

        st.cusum.weight = cfg.step_s / st.window_s
        g = st.cusum.update(m["alpha"])

        # decision state machine ---------------------------------------------------------
        if st.phase == "AWAITING_HUMAN":
            dec = self.gate.poll(st.incident)
            if dec is None:
                return
            if dec["approve"]:
                wo = self.tool_create_work_order(st.name, dec)
                self._log("create_work_order", st.name, {}, wo, f"approved by {dec['reviewer']}")
                st.phase = "ESCALATED"
            else:
                st.h_alarm *= 1.5
                self._recalibrate(st, 20.0)
                st.cusum.h = st.h_alarm
                self._log("human_rejected", st.name, dec, {"new_h_alarm": st.h_alarm, "alpha_mu": round(st.cusum.mu, 3)},
                          "reviewer rejected: re-baseline on recent data, raise threshold")
                st.phase = "NOMINAL"
            return

        if st.phase in ("NOMINAL", "WATCH") and g > cfg.h_watch and st.phase == "NOMINAL":
            st.phase = "WATCH"
            self._log("watch", st.name, {}, {"cusum": round(g, 2), "alpha": round(m["alpha"], 3)},
                      "alpha drifting below baseline -> watch")
        if st.phase == "WATCH" and g > cfg.h_suspect:
            r, dt = self._call("cross_validate", st.name, self.tool_cross_validate,
                               {"roi": st.name, "window_s": st.window_s})
            z_x = st.cusum.z(r["alpha"]) if np.isfinite(r["alpha"]) else -np.inf
            # agreement test: difference within the combined 95 % CIs, and the
            # independent estimate also below baseline
            half = lambda ci: (ci[1] - ci[0]) / 2.0  # noqa: E731
            tol = np.hypot(half(r["ci"]), half((m["ci_lo"], m["ci_hi"])))
            agree = bool(np.isfinite(r["alpha"]) and abs(r["alpha"] - m["alpha"]) <= tol and z_x > cfg.xval_z)
            r["tolerance"] = round(float(tol), 3)
            if agree:
                st.phase = "SUSPECT"
                self._log("cross_validate", st.name, {"window_s": st.window_s}, r,
                          f"independent estimator agrees (z={z_x:.1f}) -> SUSPECT", dt)
            else:
                self._log("cross_validate", st.name, {"window_s": st.window_s}, r,
                          "estimators disagree -> likely tracking artefact: relocate, do not alarm", dt)
                rr, dt2 = self._call("relocate_roi", st.name, self.tool_relocate_roi, {"roi": st.name})
                self._log("relocate_roi", st.name, {}, rr, "after failed cross-validation", dt2)
                st.cusum.g = 0.0
                st.phase = "NOMINAL"
            return
        if st.phase == "SUSPECT" and g > st.h_alarm:
            env, dt = self._call("envelope_analysis", st.name, self.tool_envelope_analysis,
                                 {"roi": st.name, "window_s": self.cfg.max_window_s})
            self._log("envelope_analysis", st.name, {}, env,
                      "impact rate matches bearing defect frequency" if env.get("match") else "no defect-frequency match", dt)
            ev, dt = self._call("render_evidence", st.name, self.tool_render_evidence,
                                {"roi": st.name, "window_s": st.window_s, "band": env.get("demod_band_hz")})
            self._log("render_evidence", st.name, {}, ev, "magnified x-t slice + spectrum for reviewer", dt)
            ev["envelope"] = env
            st.last_env_t, st.env_match = self.mon.t, bool(env.get("match"))
            pkt = self.tool_request_human_approval(st.name, ev)
            st.phase = "AWAITING_HUMAN"
            self._log("request_human_approval", st.name, {}, {"incident_id": pkt["incident_id"],
                      "recommended_action": pkt["recommended_action"]}, "human sign-off required", 0)
            return
        if st.phase in ("WATCH", "SUSPECT") and g == 0.0:
            self._log("clear", st.name, {}, {"cusum": 0.0}, "evidence vanished -> NOMINAL")
            st.phase = "NOMINAL"

    def _refine_diagnosis(self, st: RoiState) -> None:
        """Keep accumulating evidence after escalation: the spectral-shape detector
        fires before the defect frequency is identifiable, so the agent re-runs the
        envelope analysis and updates the open incident when a match appears."""
        env, dt = self._call("envelope_analysis", st.name, self.tool_envelope_analysis,
                             {"roi": st.name, "window_s": self.cfg.max_window_s})
        st.last_env_t = self.mon.t
        if env.get("match"):
            st.env_match = True
            pkt = self.gate.pending.get(st.incident)
            if pkt is not None:
                pkt["evidence"]["envelope_update"] = env
                pkt["recommended_action"] = self._recommendation({"envelope": env})
            self._log("envelope_analysis", st.name, {"follow_up": True}, env,
                      f"diagnosis refined: envelope lines match {env['nearest_defect']} "
                      f"({env['nearest_hz']} Hz) -> incident updated", dt)
        else:
            self._log("envelope_analysis", st.name, {"follow_up": True}, env, "no match yet", dt, level="debug")

    def apply_pending_decisions(self) -> None:
        """Apply reviewer decisions that arrive after the video has ended (service mode)."""
        self._poll_machine_incidents()
        for st in self.rois.values():
            if st.phase != "AWAITING_HUMAN":
                continue
            dec = self.gate.poll(st.incident)
            if dec is None:
                continue
            if dec["approve"]:
                wo = self.tool_create_work_order(st.name, dec)
                self._log("create_work_order", st.name, {}, wo, f"approved by {dec['reviewer']}")
                st.phase = "ESCALATED"
            else:
                st.h_alarm *= 1.5
                if st.cusum is not None:
                    st.cusum.h = st.h_alarm
                    st.cusum.g = 0.0
                self._log("human_rejected", st.name, dec, {"new_h_alarm": st.h_alarm},
                          "reviewer rejected: threshold raised for this ROI")
                st.phase = "NOMINAL"

    # ------------------------------------------------------------- reporting
    def summary(self) -> dict:
        return {
            "t_s": round(self.mon.t, 2),
            "rois": {n: {"phase": s.phase, "box": list(self.mon.boxes[n]), "window_s": s.window_s,
                         "alpha": s.last.get("alpha"), "baseline": None if s.cusum is None else s.cusum.mu,
                         "cusum": None if s.cusum is None else s.cusum.g, "incident": s.incident}
                     for n, s in self.rois.items()},
            "compensation": self.mon.compensate,
            "setup_invalid": self.setup_invalid,
            "incidents": list(self.gate.pending.values()),
            "structure": (self.struct.history[-1] if self.struct and self.struct.history else None),
            "noise_model": self.noise_model,
            "work_orders": self.work_orders,
            "tool_calls": sum(1 for e in self.trace),
        }

    def save_trace(self, path: str) -> None:
        with open(path, "w") as fh:
            for ev in self.trace:
                fh.write(json.dumps(ev, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
