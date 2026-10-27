"""Real-video tools: pre-flight check, floor-referenced housing motion, audio cross-check, A/B indicators.

Used by the notebook (section 6d) and by the CLI wrappers in eval/:
    check_real_video.py   -> preflight() + plot_preflight()
    analyze_fan_motion.py -> housing_motion() + plot_housing()
    real_ab.py            -> indicator_windows() + compare()

Housing motion works even when a free-standing machine walks tens of pixels:
  1. outline with cv2.HoughCircles,
  2. background (outside the outline) -> frame-to-frame homography
     (goodFeaturesToTrack + calcOpticalFlowPyrLK + findHomography/RANSAC),
  3. features on the housing -> rigid motion (estimateAffinePartial2D) in background coordinates,
  4. the clip's audio track as an independent sensor (amplitude-modulation spectrum).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import hashlib
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from .agent import AgentConfig
from .pipeline import auto_regions, expand_box, pick_reference, read_video, run_agent_session
from .spectral import SpectralConfig, analyze_window
from .tracking import track_sequence
from .spectralmatrix import relative_spectral_evidence

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
AB_KEYS = ("alpha", "tonal_db", "tone_hz", "band_ratio", "flatness", "rms", "hf_rms")


def load_frames(path: str, max_seconds=None):
    fps, gen = read_video(path, max_seconds)
    return fps, list(gen)


# ----------------------------------------------------------------------------- container facts
def probe(path: str) -> dict:
    if shutil.which("ffprobe"):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,avg_frame_rate,bit_rate,codec_name:format=duration",
                              "-of", "json", path], capture_output=True, text=True).stdout
        j = json.loads(out)
        st, fmt = j["streams"][0], j["format"]
        num, den = map(float, st["avg_frame_rate"].split("/"))
        fps = num / den
        br = float(st.get("bit_rate", 0))
        return {"codec": st["codec_name"], "width": st["width"], "height": st["height"], "fps": fps,
                "duration_s": float(fmt["duration"]), "bitrate_kbps": br / 1e3,
                "bits_per_pixel": br / (st["width"] * st["height"] * fps) if br else None}
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return {"codec": "?", "width": w, "height": h, "fps": fps, "duration_s": n / fps, "bitrate_kbps": None,
            "bits_per_pixel": None}


# ----------------------------------------------------------------------------- pre-flight
def stabilise(frames):
    """Chain frame-to-frame RANSAC homographies back to frame 0."""
    H, W = frames[0].shape
    out, prev, Hacc, inl = [frames[0]], frames[0], np.eye(3), []
    for cur in frames[1:]:
        pp = cv2.goodFeaturesToTrack(prev, 1500, 0.01, 6)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pp, None, winSize=(21, 21), maxLevel=4)
        ok = st.ravel() == 1
        Hk, mask = cv2.findHomography(nxt[ok], pp[ok], cv2.RANSAC, 1.0)
        Hacc = Hacc @ Hk
        inl.append(float(mask.mean()))
        out.append(cv2.warpPerspective(cur, Hacc, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))
        prev = cur
    return out, float(np.mean(inl))


def preflight(path: str, frames=None, fps=None, keep_series: bool = False) -> dict:
    """Can this footage support sub-pixel vibrometry? Includes the agent's own setup verdict."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # empty windows on very short clips
        return _preflight(path, frames, fps, keep_series)


def _preflight(path, frames, fps, keep_series):
    meta = probe(path)
    if frames is None:
        fps, frames = load_frames(path)
    regions = auto_regions(frames[0], n=5, size=56)
    raw, _, _ = track_sequence(frames, regions, "ic")
    rms = lambda d: {k: float(np.sqrt(np.sum((v - v.mean(axis=1, keepdims=True)).var(axis=1)))) for k, v in d.items()}  # noqa: E731
    stab, inliers = stabilise(frames)
    st_disp, _, _ = track_sequence(stab, regions, "ic")
    rois = dict(regions)
    ref, _ = pick_reference(frames, fps, rois, seconds=min(3.0, len(frames) / fps / 2))
    agent = run_agent_session(iter(frames), fps, rois, {k: expand_box(b, frames[0].shape) for k, b in rois.items()},
                              ref, cfg=AgentConfig(window_s=6, max_window_s=8, calib_s=8))
    res = {"file": os.path.basename(path), **meta, "frames": len(frames), "reference_box": list(ref),
           "raw_motion_rms_px": rms(raw), "after_stabilisation_rms_px": rms(st_disp),
           "stabilisation_inlier_ratio": inliers,
           "agent_verdict": agent.setup_invalid or {"problems": [], "note": "setup accepted"},
           "agent_decisions": [{k: e[k] for k in ("t", "tool", "decision")} for e in agent.trace if e["level"] != "debug"]}
    if keep_series:
        res["_series"] = {"frame0": frames[0], "raw": raw, "stab": st_disp, "regions": regions, "fps": fps}
    return res


# ----------------------------------------------------------------------------- housing motion + audio
def audio_modulation(path: str, band=(200.0, 2000.0), sr: int = 8000):
    """Amplitude-modulation spectrum of the clip's audio (None if no audio / no ffmpeg)."""
    if not shutil.which("ffmpeg"):
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(float) / 32768
    if x.size < sr:
        return None
    env = np.abs(hilbert(sosfiltfilt(butter(4, list(band), btype="band", fs=sr, output="sos"), x)))
    fa, Pa = welch(env - env.mean(), fs=sr, nperseg=min(32768, len(env)))
    m = (fa > 3) & (fa < 60)
    return fa[m], Pa[m], float(x.std())


def housing_motion(path: str, frames=None, fps=None) -> dict:
    if frames is None:
        fps, frames = load_frames(path)
    fr = frames
    H, W = fr[0].shape
    g = cv2.GaussianBlur(fr[0], (0, 0), 2)
    circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=400, param1=120, param2=60,
                            minRadius=int(0.25 * H), maxRadius=int(0.6 * H))
    if circ is None:
        raise RuntimeError("machine outline not found (HoughCircles)")
    cx, cy, r = map(float, circ[0][0])
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(xx - cx, yy - cy)
    m_bg = (dist > 1.15 * r).astype(np.uint8) * 255
    m_fan = ((dist < 0.98 * r) & (dist > 0.55 * r)).astype(np.uint8) * 255
    Hacc, A_acc = np.eye(3), np.eye(3)
    cam, pose = [], []
    for k in range(1, len(fr)):
        prev, cur = fr[k - 1], fr[k]
        p = cv2.goodFeaturesToTrack(prev, 1200, 0.01, 7, mask=m_bg)
        q, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p, None, winSize=(21, 21), maxLevel=4)
        ok = st.ravel() == 1
        cam.append(np.median((q - p).reshape(-1, 2)[ok], axis=0))
        Hk, _ = cv2.findHomography(q[ok], p[ok], cv2.RANSAC, 1.0)          # cur -> prev (background plane)
        H_prev = Hacc.copy()
        Hacc = Hacc @ Hk                                                    # cur -> frame 0
        pf = cv2.goodFeaturesToTrack(prev, 800, 0.01, 5, mask=m_fan)
        qf, stf, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pf, None, winSize=(21, 21), maxLevel=4)
        okf = stf.ravel() == 1
        a = cv2.perspectiveTransform(pf[okf].reshape(-1, 1, 2), H_prev)
        b = cv2.perspectiveTransform(qf[okf].reshape(-1, 1, 2), Hacc)
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=0.7)
        A_acc = np.vstack([M, [0, 0, 1]]) @ A_acc
        pose.append((A_acc[0, 2], A_acc[1, 2], np.degrees(np.arctan2(A_acc[1, 0], A_acc[0, 0])), float(inl.mean())))
    P = np.array(pose)
    rot = lambda x, y, a: np.array([[np.cos(np.radians(a)), -np.sin(np.radians(a)), x],  # noqa: E731
                                    [np.sin(np.radians(a)), np.cos(np.radians(a)), y], [0, 0, 1]])
    cen = np.array([(rot(x, y, a) @ np.array([cx, cy, 1.0]))[:2] for x, y, a, _ in P])
    d = cen - cen[0]
    v = np.diff(cen, axis=0)
    hp = v - np.array([np.convolve(v[:, i], np.ones(9) / 9, "same") for i in range(2)]).T   # > ~3 Hz
    nper = min(128, len(hp))
    f, Pv = welch(hp.T, fs=fps, nperseg=nper, axis=-1)
    Pv = Pv.sum(0)
    cam = np.array(cam)
    fc, Pc = welch(cam.T, fs=fps, nperseg=nper, axis=-1)
    Pc = Pc.sum(0)
    sel = (f > 3) & (f < fps / 2 - 0.5)
    peak_v = float(f[sel][np.argmax(Pv[sel])])
    lo, hi = max(1.0, peak_v - 0.9), min(fps / 2 - 0.2, peak_v + 0.9)
    tone_px = float(np.sqrt((sosfiltfilt(butter(4, [lo, hi], btype="band", fs=fps, output="sos"), d, axis=0) ** 2).sum(1).mean()))
    nb = sel & ((f < lo - 0.3) | (f > hi + 0.3))
    prom_v = float(10 * np.log10(Pv[(f >= lo) & (f <= hi)].max() / np.median(Pv[nb])))
    au = audio_modulation(path)
    if au is not None:
        fa, Pa, arms = au
        la = (fa > peak_v - 0.6) & (fa < peak_v + 0.6)
        peak_a = float(fa[np.argmax(Pa)])
        prom_a = float(10 * np.log10(Pa[la].max() / np.median(Pa[(fa > 3) & (fa < 30)]))) if la.any() else None
    else:
        fa = Pa = np.array([])
        peak_a = prom_a = arms = None
    return {"file": os.path.basename(path), "fps": fps, "frames": len(fr), "circle": [cx, cy, r],
            "drift_px": d[-1].tolist(), "rotation_deg": float(P[-1, 2]), "housing_inliers": float(P[:, 3].mean()),
            "vibration_rms_px_per_frame": float(np.sqrt((hp ** 2).sum(1).mean())),
            "camera_motion_rms_px_per_frame": float(np.sqrt((cam ** 2).sum(1).mean())),
            "video_peak_hz": peak_v, "camera_peak_hz": float(fc[sel][np.argmax(Pc[sel])]),
            "audio_modulation_peak_hz": peak_a, "audio_rms": arms,
            "tone_hz": peak_v, "tone_displacement_rms_px": tone_px, "tone_prominence_db_video": prom_v,
            "tone_prominence_db_audio_at_same_hz": prom_a, "outline_radius_px": r,
            "_plot": {"t": (np.arange(1, len(fr)) / fps).tolist(), "d": d.tolist(), "rot": P[:, 2].tolist(),
                      "f": f.tolist(), "Pv": Pv.tolist(), "fc": fc.tolist(), "Pc": Pc.tolist(),
                      "fa": np.asarray(fa).tolist(), "Pa": np.asarray(Pa).tolist(), "frame0": fr[0]}}


def public(res: dict) -> dict:
    """Result without plotting payloads (JSON-safe)."""
    return {k: v for k, v in res.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------- A/B indicators
def indicator_windows(motion: dict, win_s: float = 6.0, step_s: float = 1.0) -> List[dict]:
    """All vibrovision.spectral indicators on sliding windows of the housing trajectory."""
    fps = motion["fps"]
    d = np.asarray(motion["_plot"]["d"]).T
    cfg = SpectralConfig.for_fps(fps)
    w, s = int(round(win_s * fps)), int(round(step_s * fps))
    rows = []
    for i in range(0, d.shape[1] - w + 1, s):
        out = analyze_window(d[:, i:i + w], cfg, 0.0)
        rows.append({k: out.get(k) for k in AB_KEYS} | {"t": (i + w / 2) / fps})
    return rows


def summarize(rows: List[dict]) -> Dict[str, Optional[dict]]:
    out = {}
    for k in AB_KEYS:
        v = np.array([x[k] for x in rows], float)
        v = v[np.isfinite(v)]
        out[k] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(v.size)} if v.size else None
    return out


def compare(sum_a: dict, sum_b: dict) -> Dict[str, bool]:
    """Indicators whose window ranges do not overlap between the two clips."""
    return {k: bool(sum_a[k] and sum_b[k] and (sum_a[k]["max"] < sum_b[k]["min"] or sum_b[k]["max"] < sum_a[k]["min"]))
            for k in AB_KEYS}


def compare_clips(paths, labels=("Sin moneda", "Con moneda"), out_figure=None, max_seconds=30):
    """Fresh A/B analysis shared by service, CLI and the standalone notebook.

    One housing's x/y channels are not separate machine parts. These short clips
    support spectral evidence, not identification of a complete stiffness matrix.
    """
    if len(paths) != 2 or len(labels) != 2:
        raise ValueError("provide exactly two clips and two labels")
    motions, clips, signals = [], [], []
    for path, label in zip(paths, labels):
        fps, frames = load_frames(str(path), max_seconds=max_seconds)
        if len(frames) < 8 * fps:
            raise ValueError("A/B analysis requires at least 8 seconds per clip")
        if max(frames[0].shape) > 1280:
            raise ValueError("A/B clips must be <=1280 pixels per side; export a smaller analysis copy")
        motion = housing_motion(str(path), frames, fps)
        series = np.asarray(motion["_plot"]["d"]).T
        signals.append(series)
        summary = summarize(indicator_windows(motion, 6, 1))
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        clips.append({"label": label, "sha256": digest.hexdigest(), "file": os.path.basename(path),
                      "fps": fps, "analyzed_seconds": len(frames) / fps,
                      "summary": summary, "motion": public(motion)})
        motions.append(motion)
        del frames
    joint = relative_spectral_evidence(signals[0], signals[1], clips[0]["fps"], clips[1]["fps"])
    if out_figure:
        plot_housing(motions, labels, str(out_figure))
    ratio = clips[1]["motion"]["tone_displacement_rms_px"] / max(clips[0]["motion"]["tone_displacement_rms_px"], 1e-15)
    return {"schema_version": "3.0", "source": "fresh_analysis", "clips": clips,
            "range_separation": compare(clips[0]["summary"], clips[1]["summary"]),
            "tone_amplitude_ratio": float(ratio), "joint_spectral": joint,
            "structural_verdict": {"decision": "abstain", "reason": "Only one housing is measured and clips are shorter than a structural window. No complete machine stiffness is identified."},
            "explanation": "El segundo video cambia el movimiento y los tonos del ventilador. Esto no demuestra desgaste temprano ni identifica todos sus anclajes.",
            "limits": ["one short recording per condition", "overlapping windows are not independent trials",
                       "no contact-sensor ground truth", "audio is a corroborating channel, not calibrated ground truth",
                       "joint spectrum compares housing x/y channels, not two independent parts",
                       "range separation is descriptive, not a significance test"],
            "artifact": os.path.basename(out_figure) if out_figure else None}


# ----------------------------------------------------------------------------- figures (Figure API: no backend switch)
def plot_housing(results: Sequence[dict], labels: Sequence[str], out: Optional[str] = None):
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    n = len(results)
    fig = Figure(figsize=(14, 3.4 * n + 0.6), dpi=110)
    gs = fig.add_gridspec(n, 3, width_ratios=[1.0, 1.1, 1.4])
    share2 = share3 = None                 # same y-scales on every row so clips compare at a glance
    for i, (r, lab) in enumerate(zip(results, labels)):
        p = r["_plot"]
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(p["frame0"], cmap="gray"); ax.axis("off")
        cx, cy, rr = r["circle"]
        ax.add_patch(Circle((cx, cy), rr, fill=False, ec=C[1], lw=2))
        d = np.array(p["d"])
        ax.plot(cx + d[:, 0], cy + d[:, 1], color=C[0], lw=2)
        ax.set_title(f"{r['file']} ({lab})\noutline (HoughCircles) + centre path", fontsize=9)
        ax2 = fig.add_subplot(gs[i, 1], sharey=share2)
        share2 = share2 or ax2
        t = np.array(p["t"])
        ax2.plot(t, d[:, 0], color=C[0], label="x drift (px)")
        ax2.plot(t, d[:, 1], color=C[2], label="y drift (px)")
        ax2.plot(t, p["rot"], color=C[1], label="rotation (deg)")
        ax2.set_xlabel("s"); ax2.legend(fontsize=7); ax2.grid(alpha=.25)
        ax2.set_title(f"housing vs floor: drift {np.hypot(*r['drift_px']):.0f} px, rotation {r['rotation_deg']:+.1f} deg", fontsize=9)
        ax3 = fig.add_subplot(gs[i, 2], sharey=share3)
        share3 = share3 or ax3
        f, Pv, fc, Pc = map(np.array, (p["f"], p["Pv"], p["fc"], p["Pc"]))
        ax3.semilogy(f, Pv, color=C[0], lw=2, label=f"housing vibration (video), peak {r['video_peak_hz']:.1f} Hz")
        ax3.semilogy(fc, Pc, color="#8a8984", lw=1.2, label=f"camera motion (background), peak {r['camera_peak_hz']:.1f} Hz")
        handles, labs = ax3.get_legend_handles_labels()
        if len(p["fa"]):
            ax3b = ax3.twinx()
            ax3b.plot(p["fa"], np.array(p["Pa"]) / max(p["Pa"]), color=C[1], lw=1.0, alpha=.8,
                      label=f"audio modulation (normalised), peak {r['audio_modulation_peak_hz']:.1f} Hz")
            ax3b.set_yticks([]); ax3b.set_ylim(0, 1.6)
            h2, l2 = ax3b.get_legend_handles_labels()
            handles, labs = handles + h2, labs + l2
        ax3.set_xlim(0, 40); ax3.axvline(r["fps"] / 2, color="#52514e", ls=":", lw=1)
        ax3.text(r["fps"] / 2 + 0.3, 1e-5, "video Nyquist", fontsize=7, color="#52514e")
        ax3.set_xlabel("Hz"); ax3.grid(alpha=.25)
        ax3.legend(handles, labs, fontsize=7, loc="upper right")
        ax3.set_title("spectra: housing (video) vs camera vs microphone", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out)
    return fig


def plot_preflight(results: Sequence[dict], out: Optional[str] = None):
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    n = len(results)
    fig = Figure(figsize=(14, 3.6 * n), dpi=110)
    axs = fig.subplots(n, 3, squeeze=False, gridspec_kw={"width_ratios": [1.1, 1.3, 1.3]})
    for row, res in enumerate(results):
        s = res["_series"]
        ax = axs[row, 0]
        ax.imshow(s["frame0"], cmap="gray"); ax.axis("off")
        for i, (k, (x, y, w, h)) in enumerate(s["regions"].items()):
            ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=C[i % 5], lw=2))
        x, y, w, h = res["reference_box"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="white", ls="--", lw=1.5))
        ax.set_title(f"{res['file']}: {res['width']}x{res['height']} @ {res['fps']:.1f} fps, {res['duration_s']:.0f} s", fontsize=9)
        t = np.arange(s["raw"][next(iter(s["raw"]))].shape[1]) / s["fps"]
        for i, k in enumerate(s["regions"]):
            axs[row, 1].plot(t, s["raw"][k][0] - s["raw"][k][0, 0], color=C[i % 5], lw=1.2, label=k)
            axs[row, 2].plot(t, s["stab"][k][0] - s["stab"][k][0].mean(), color=C[i % 5], lw=1.0)
        axs[row, 1].set_title("raw x-motion of 5 patches (px)", fontsize=9)
        axs[row, 2].set_title("after homography stabilisation (px)", fontsize=9)
        for a in axs[row, 1:]:
            a.set_xlabel("s"); a.grid(alpha=.25)
        axs[row, 2].axhspan(-0.05, 0.05, color="#1baf7a", alpha=.25, label="needed: ~0.01-0.05 px")
        axs[row, 2].legend(fontsize=7, loc="upper right")
        axs[row, 1].legend(fontsize=7, ncol=5, loc="upper left")
    fig.tight_layout()
    if out:
        fig.savefig(out)
    return fig
