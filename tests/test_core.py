import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibrovision.agent import AgentConfig, BearingSpec, HumanGate, InspectionAgent  # noqa: E402
from vibrovision.detect import Cusum, first_alarm  # noqa: E402
from vibrovision.monitor import StreamingMonitor  # noqa: E402
from vibrovision.pipeline import build_scenario  # noqa: E402
from vibrovision.simulate import (FourierShifter, build_canvas, colored_noise, default_scene,  # noqa: E402
                                  generate_displacements, render_frames)
from vibrovision.spectral import SpectralConfig, analyze_window, defect_line_scores, envelope_spectrum  # noqa: E402
from vibrovision.tracking import make_tracker  # noqa: E402

FS = 120.0


def _shifted_frames(shifts):
    sc = default_scene(duration_s=1, cam_noise=0.0)
    canvas, spr = build_canvas(sc)
    c = sc.components[0]
    x, y, w, h = c.region
    fsh = FourierShifter(spr[c.name])
    out = []
    for dx, dy in shifts:
        f = canvas.copy()
        f[y:y + h, x:x + w] = fsh.shift(dx, dy)[8:8 + h, 8:8 + w]
        out.append(np.clip(np.rint(f), 0, 255).astype(np.uint8))
    return out, c.roi


@pytest.mark.parametrize("method,tol", [("ic", 0.01), ("ecc", 0.02), ("lk", 0.03)])
def test_trackers_recover_subpixel_shift(method, tol):
    shifts = [(0, 0), (0.05, -0.02), (0.2, 0.1), (-0.3, 0.25), (0.6, -0.4)]
    frames, roi = _shifted_frames(shifts)
    tr = make_tracker(method, frames[0], roi)
    est = np.array([tr.track(f)[:2] for f in frames[1:]])
    err = np.abs(est - np.array(shifts[1:]))
    assert err.max() < tol * 3 and err.mean() < tol


def test_alpha_recovers_power_law_and_is_gain_invariant():
    rng = np.random.default_rng(0)
    cfg = SpectralConfig(fs=FS)
    for alpha in (0.8, 1.6):
        xy = np.vstack([colored_noise(1440, alpha, FS, rng) for _ in range(2)])
        a1 = analyze_window(xy, cfg)["alpha"]
        a2 = analyze_window(7.5 * xy, cfg)["alpha"]
        assert abs(a1 - alpha) < 0.15
        assert abs(a1 - a2) < 1e-6        # log-log slope ignores the px->mm scale


def test_tones_do_not_bias_alpha():
    rng = np.random.default_rng(1)
    t = np.arange(1440) / FS
    xy = np.vstack([colored_noise(1440, 1.6, FS, rng) for _ in range(2)])
    tone = 0.8 * np.sin(2 * np.pi * 24.6 * t)
    cfg = SpectralConfig(fs=FS)
    assert abs(analyze_window(xy + tone, cfg)["alpha"] - analyze_window(xy, cfg)["alpha"]) < 0.1


def test_noise_floor_correction_removes_flattening():
    rng = np.random.default_rng(2)
    xy = 0.2 * np.vstack([colored_noise(1440, 1.6, FS, rng) for _ in range(2)])
    sigma = 0.01
    noisy = xy + rng.normal(0, sigma, xy.shape)
    N = 2 * sigma ** 2 / (FS / 2)
    cfg = SpectralConfig(fs=FS)
    a_true = analyze_window(xy, cfg)["alpha"]
    a_raw = analyze_window(noisy, cfg)["alpha"]
    a_cor = analyze_window(noisy, cfg, noise_psd=N)["alpha"]
    assert abs(a_cor - a_true) < abs(a_raw - a_true)


def test_envelope_finds_impact_rate():
    sc = default_scene(seed=3)
    sc.component("Rodamiento_B").fault.onset_s = 0
    sc.component("Rodamiento_B").fault.full_s = 1
    _, truth, _ = generate_displacements(sc)
    xy = truth["Rodamiento_B"][:, -int(18 * FS):]
    env = envelope_spectrum(xy, SpectralConfig(fs=FS), band=(34, 52))
    cands = {k: v for k, v in BearingSpec(shaft_hz=2.05).defect_frequencies().items() if 2 <= v <= 20}
    scores = defect_line_scores(env, cands)
    assert max(scores, key=scores.get) == "BPFO" and scores["BPFO"] >= 8


def test_bearing_frequencies():
    f = BearingSpec(shaft_hz=2.05).defect_frequencies()
    assert abs(f["BPFO"] - 7.29) < 0.02 and f["BPFI"] > f["BPFO"]


def test_cusum_alarms_on_shift_only():
    rng = np.random.default_rng(4)
    v = 1.6 + 0.04 * rng.standard_normal(200)
    cal = np.arange(200) < 50
    t, _ = first_alarm(np.arange(200), v, cal, "down", h=6, sd_floor=0.04, weight=0.25)
    assert t is None
    v[120:] -= 0.2
    t, _ = first_alarm(np.arange(200), v, cal, "down", h=6, sd_floor=0.04, weight=0.25)
    assert t is not None and 120 <= t < 160


def test_agent_relocates_bad_roi_and_gate_flow():
    sc = build_scenario("fault_bad_roi", seed=5, duration_s=6.0)
    _, truth, shake = generate_displacements(sc)
    rois = {c.name: c.roi for c in sc.components}
    regions = {c.name: c.region for c in sc.components}
    mon = StreamingMonitor(sc.fps, rois, sc.reference_roi, buffer_s=6)
    agent = InspectionAgent(mon, regions, SpectralConfig(fs=sc.fps), AgentConfig(window_s=4, max_window_s=4))
    for fr in render_frames(sc, truth, shake):
        mon.push(fr)
        if agent.due() and mon.n > 1:
            agent.step()
    moved = [e for e in agent.trace if e["tool"] == "relocate_roi"]
    assert moved and moved[0]["roi"] == "Rodamiento_B"
    assert mon.boxes["Rodamiento_B"] != sc.component("Rodamiento_B").roi
    # human gate: queue mode, decision applied later
    gate = HumanGate()
    gate.request({"incident_id": "X", "component": "c"})
    assert gate.poll("X") is None
    gate.decide("X", True, "tester")
    assert gate.poll("X")["approve"] and gate.pending["X"]["status"] == "APPROVED"


def test_render_is_deterministic():
    sc = default_scene(duration_s=0.2)
    _, truth, shake = generate_displacements(sc)
    a = list(render_frames(sc, truth, shake))
    b = list(render_frames(sc, truth, shake))
    assert all((x == y).all() for x, y in zip(a, b))


def test_ic_photometric_ignores_exposure_change():
    frames, roi = _shifted_frames([(0, 0), (0.2, -0.1)])
    bright = np.clip(frames[1].astype(np.float32) * 1.15 + 6, 0, 255).astype(np.uint8)
    from vibrovision.tracking import ICTracker
    d_photo = ICTracker(frames[0], roi).track(bright)[:2]
    d_raw = ICTracker(frames[0], roi, photometric=False).track(bright)[:2]
    assert np.allclose(d_photo, (0.2, -0.1), atol=0.02)
    assert np.abs(np.array(d_raw) - (0.2, -0.1)).max() > np.abs(np.array(d_photo) - (0.2, -0.1)).max()


def test_surrogate_reproduces_error_spectrum():
    from vibrovision.surrogate import colored_error
    from vibrovision.spectral import psd
    f = np.linspace(0, 60, 129)
    P = 1e-5 * (1 + 3 * (f < 10))
    e = colored_error(120 * 600, FS, f, P, np.random.default_rng(0))
    fe, Pe = psd(e, SpectralConfig(fs=FS))
    lo, hi = Pe[(fe > 3) & (fe < 8)].mean(), Pe[(fe > 30) & (fe < 50)].mean()
    assert abs(hi / 1e-5 - 1) < 0.15 and abs(lo / hi - 4) < 0.8


def test_production_profile():
    c = AgentConfig.production()
    assert c.calib_s >= 600 and c.h_alarm >= 10 and c.step_s == 2.0


def test_h264_roundtrip(tmp_path):
    from vibrovision.simulate import h264_roundtrip
    sc = default_scene(duration_s=0.5)
    _, truth, shake = generate_displacements(sc)
    out = list(h264_roundtrip(render_frames(sc, truth, shake), sc.fps, 23, str(tmp_path)))
    assert len(out) == sc.n_frames and out[0].shape == (sc.height, sc.width)


def test_agent_rejects_handheld_footage():
    sc = build_scenario("healthy", seed=9, duration_s=4.0, shake_rms_px=4.0, shake_band_hz=(0.3, 3.0))
    _, truth, shake = generate_displacements(sc)
    rois = {c.name: c.roi for c in sc.components}
    regions = {c.name: c.region for c in sc.components}
    mon = StreamingMonitor(sc.fps, rois, sc.reference_roi, buffer_s=4)
    agent = InspectionAgent(mon, regions, SpectralConfig(fs=sc.fps), AgentConfig(window_s=2, max_window_s=2))
    for fr in render_frames(sc, truth, shake):
        mon.push(fr)
        if agent.due() and mon.n > 1:
            agent.step()
    assert agent.setup_invalid is not None
    assert all(st.phase == "INVALID_SETUP" for st in agent.rois.values())
    assert not agent.gate.pending                        # no diagnosis from invalid footage


def test_fractional_fps_buffer():
    mon = StreamingMonitor(29.87, {"a": (10, 10, 20, 20)}, None, buffer_s=2)
    for _ in range(5):
        mon.push(np.zeros((60, 60), np.uint8))
    assert mon.buffered(1.0 / 29.87).shape[0] == 1


def test_tonal_prominence_flags_imbalance_tone_and_is_gain_invariant():
    rng = np.random.default_rng(3)
    fs = 29.9
    cfg = SpectralConfig.for_fps(fs)
    n = int(6 * fs)
    t = np.arange(n) / fs
    bb = np.vstack([colored_noise(n, 1.5, fs, rng) for _ in range(2)])
    tone = np.vstack([2.0 * np.sin(2 * np.pi * 12.4 * t), 1.5 * np.cos(2 * np.pi * 12.4 * t)])
    a = analyze_window(bb, cfg)
    b = analyze_window(bb + tone, cfg)
    c = analyze_window(5 * (bb + tone), cfg)
    assert b["tonal_db"] > a["tonal_db"] + 5 and abs(b["tone_hz"] - 12.4) < 0.5
    assert abs(b["tonal_db"] - c["tonal_db"]) < 1e-6


def test_spectral_config_scales_with_fps():
    assert SpectralConfig.for_fps(120.0) == SpectralConfig(fs=120.0)
    c = SpectralConfig.for_fps(29.9)
    assert c.fit_band[1] < 14.95 and c.hi_band[1] < 14.95 and c.nperseg == 64
