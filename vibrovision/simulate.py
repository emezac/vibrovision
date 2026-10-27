"""Physically-motivated synthetic test bench for camera-based vibrometry.

What is simulated (and why it is a *test bench*, not proof of field performance):

* Each machine component moves rigidly in the image plane (sub-pixel, 2 axes).
* Healthy motion = broadband structural response (power-law spectrum, high-passed
  so there is no unbounded drift) + rotation tones (1x, 2x shaft speed).
* Fault = a train of small impacts at a defect repetition rate (e.g. a bearing
  outer-race frequency) whose energy rings a high-frequency structural mode.
  Severity grows exponentially between ``onset_s`` and ``full_s``. Impacts add
  little total energy (RMS barely changes) but concentrate it at high
  frequency, where the healthy spectrum is weak: the spectral *shape* changes
  long before the amplitude does.
* Camera nuisances: sensor noise + 8-bit quantisation, global camera shake,
  illumination flicker (aliased mains flicker), a mid-run zoom/gain change, and
  contrast loss.

Ground-truth displacements are returned so every downstream number can be
checked against truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import lfilter

Box = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class FaultSpec:
    onset_s: float = 70.0          # degradation starts
    full_s: float = 170.0          # severity reaches 1.0
    impact_rate_hz: float = 7.3    # defect repetition rate (e.g. BPFO)
    ring_hz: float = 41.0          # structural mode excited by each impact
    ring_zeta: float = 0.04        # damping ratio of that mode
    impact_rms_px: float = 0.075   # RMS of the impact response at full severity
    growth: float = 4.0            # exponential growth rate (0 = linear)
    jitter: float = 0.02           # slip / jitter of impact timing


@dataclass
class ComponentSpec:
    name: str
    region: Box                     # where the part is in the image (user's coarse box)
    roi: Box                        # initial measurement ROI (may be badly placed)
    broadband_rms_px: float = 0.22
    broadband_alpha: float = 1.6
    tones: List[Tuple[float, float]] = field(default_factory=lambda: [(24.6, 0.05), (49.2, 0.02)])
    fault: Optional[FaultSpec] = None


@dataclass
class SceneSpec:
    fps: float = 120.0
    duration_s: float = 180.0
    width: int = 320
    height: int = 240
    components: List[ComponentSpec] = field(default_factory=list)
    reference_roi: Box = (130, 186, 56, 44)   # static speckle target on the floor
    cam_noise: float = 2.0         # sensor noise, grey levels (std)
    shake_rms_px: float = 0.0      # global camera shake
    shake_band_hz: Tuple[float, float] = (0.5, 6.0)
    flicker_amp: float = 0.0       # relative illumination modulation
    flicker_hz: float = 100.0      # mains flicker (aliases at camera fps)
    zoom_at_s: Optional[float] = None
    zoom_gain: float = 1.0         # displacement gain after zoom (camera moved closer)
    contrast: float = 1.0          # texture contrast multiplier (lighting quality)
    # operating-condition variability (slow, minutes): 0 disables (keeps old scenarios identical)
    speed_var: float = 0.0         # relative shaft-speed excursion (tones move with it)
    load_var: float = 0.0          # relative excursion of broadband vibration amplitude with load
    alpha_load_coupling: float = 0.0  # +/- change of the healthy alpha with load (sensitivity test)
    # camera artefacts
    rolling_shutter_s: float = 0.0  # sensor readout time top->bottom (0 = global shutter)
    exposure_var: float = 0.0      # auto-exposure: relative gain excursions (steps + drift)
    structure: Optional[object] = None  # vibrovision.structure.StructureSpec: coupled parts (M, C, K_g + L_W)
    seed: int = 7

    @property
    def n_frames(self) -> int:
        return int(round(self.fps * self.duration_s))

    def component(self, name: str) -> ComponentSpec:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)


def default_scene(fault: bool = True, **overrides) -> SceneSpec:
    """Four-part machine (motor, gearbox, two bearing housings) + static target."""
    comps = [
        ComponentSpec("Motor", region=(12, 40, 74, 92), roi=(22, 58, 48, 48),
                      broadband_rms_px=0.24, tones=[(24.6, 0.07), (49.2, 0.025)]),
        ComponentSpec("Caja_Engranajes", region=(92, 44, 72, 84), roi=(104, 60, 48, 48),
                      broadband_rms_px=0.20, tones=[(24.6, 0.04), (36.9, 0.035)]),
        ComponentSpec("Rodamiento_A", region=(172, 52, 64, 68), roi=(180, 62, 44, 44),
                      broadband_rms_px=0.18),
        ComponentSpec("Rodamiento_B", region=(244, 52, 64, 68), roi=(252, 62, 44, 44),
                      broadband_rms_px=0.18, fault=FaultSpec() if fault else None),
    ]
    scene = SceneSpec(components=comps)
    return replace(scene, **overrides)


# --------------------------------------------------------------------------- signals

def colored_noise(n: int, alpha: float, fs: float, rng: np.random.Generator,
                  f_lo: float = 0.8) -> np.ndarray:
    """Unit-variance noise with S(f) ~ f^-alpha above f_lo, flat below (no drift)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    shape = np.maximum(f, f_lo) ** (-alpha / 2.0)
    shape[0] = 0.0
    x = np.fft.irfft(spec * shape, n)
    return x / max(x.std(), 1e-12)


def _resonator(fs: float, f0: float, zeta: float):
    """Discrete 2nd-order resonator (impulse-invariant damped sinusoid)."""
    w = 2 * np.pi * f0 / fs
    r = np.exp(-zeta * w)
    b = [0.0, r * np.sin(w)]
    a = [1.0, -2 * r * np.cos(w), r * r]
    return b, a


def severity_profile(t: np.ndarray, fault: FaultSpec) -> np.ndarray:
    u = np.clip((t - fault.onset_s) / (fault.full_s - fault.onset_s), 0.0, 1.0)
    if fault.growth <= 0:
        return u
    return (np.exp(fault.growth * u) - 1.0) / (np.exp(fault.growth) - 1.0)


def impact_response(t: np.ndarray, fs: float, fault: FaultSpec,
                    rng: np.random.Generator) -> np.ndarray:
    """Impact train at the defect rate, ringing a structural mode; unit RMS at full severity."""
    n = len(t)
    imp = np.zeros(n)
    k, tk = 0, rng.uniform(0, 1.0 / fault.impact_rate_hz)
    while tk < t[-1]:
        idx = int(round(tk * fs))
        if idx < n:
            imp[idx] += 1.0 + 0.25 * rng.standard_normal()
        tk += (1.0 / fault.impact_rate_hz) * (1.0 + fault.jitter * rng.standard_normal())
        k += 1
    b, a = _resonator(fs, fault.ring_hz, fault.ring_zeta)
    ring = lfilter(b, a, imp)
    ring /= max(ring.std(), 1e-12)          # unit RMS for a fully developed fault
    return ring * severity_profile(t, fault)


def slow_process(t: np.ndarray, rng: np.random.Generator, periods=(180.0, 1800.0), k: int = 3) -> np.ndarray:
    """Smooth operating-condition trajectory in [-1, 1] with periods of minutes."""
    x = sum(np.sin(2 * np.pi * t / rng.uniform(*periods) + rng.uniform(0, 2 * np.pi)) for _ in range(k))
    return x / max(np.abs(x).max(), 1e-12)


def operating_conditions(scene: SceneSpec, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(speed factor, load factor, load state in [-1,1]); separate RNG stream."""
    op = np.random.default_rng(scene.seed + 11)
    s = slow_process(t, op)
    l = slow_process(t, op)
    return 1.0 + scene.speed_var * s, 1.0 + scene.load_var * l, l


def generate_displacements(scene: SceneSpec) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Return t, {component: (2, n) px}, shake (2, n) px. Truth for every frame."""
    rng = np.random.default_rng(scene.seed)
    n, fs = scene.n_frames, scene.fps
    t = np.arange(n) / fs
    if scene.structure is not None:
        from .structure import simulate_structure        # lazy: structure imports this module
        _, out, _ = simulate_structure(scene.structure, fs, scene.duration_s, seed=scene.seed)
        return t, out, _shake(scene, rng, t)
    varying = scene.speed_var > 0 or scene.load_var > 0 or scene.alpha_load_coupling > 0
    speed, load, lstate = operating_conditions(scene, t) if varying else (1.0, 1.0, 0.0)
    op_rng = np.random.default_rng(scene.seed + 13)
    out: Dict[str, np.ndarray] = {}
    for comp in scene.components:
        xy = np.zeros((2, n))
        for ax in range(2):
            axis_gain = 1.0 if ax == 0 else 0.7      # anisotropic structure
            bb = colored_noise(n, comp.broadband_alpha, fs, rng)
            if scene.alpha_load_coupling > 0:
                # healthy spectral slope itself drifts with load: mix steeper/flatter versions
                d = scene.alpha_load_coupling
                steep = colored_noise(n, comp.broadband_alpha + d, fs, op_rng)
                flat = colored_noise(n, comp.broadband_alpha - d, fs, op_rng)
                w = (1.0 + lstate) / 2.0               # high load -> flatter
                bb = np.sqrt(1 - w) * steep + np.sqrt(w) * flat
            sig = comp.broadband_rms_px * load * bb
            for f0, amp in comp.tones:
                wander = 1.0 + 0.002 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
                phase = 2 * np.pi * np.cumsum(f0 * wander * speed) / fs + rng.uniform(0, 2 * np.pi)
                sig += np.sqrt(2) * amp * np.sin(phase)
            if comp.fault is not None:
                sig += comp.fault.impact_rms_px * impact_response(t, fs, comp.fault, rng)
            xy[ax] = axis_gain * sig
        out[comp.name] = xy

    shake = _shake(scene, rng, t)

    if scene.zoom_at_s is not None and scene.zoom_gain != 1.0:
        g = np.where(t >= scene.zoom_at_s, scene.zoom_gain, 1.0)
        out = {k: v * g for k, v in out.items()}
    return t, out, shake


def _shake(scene: SceneSpec, rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    n, fs = len(t), scene.fps
    shake = np.zeros((2, n))
    if scene.shake_rms_px > 0:
        lo, hi = scene.shake_band_hz
        for ax in range(2):
            spec = np.fft.rfft(rng.standard_normal(n))
            f = np.fft.rfftfreq(n, 1.0 / fs)
            spec[(f < lo) | (f > hi)] = 0.0
            s = np.fft.irfft(spec, n)
            shake[ax] = scene.shake_rms_px * s / max(s.std(), 1e-12)
    return shake


# --------------------------------------------------------------------------- rendering

def _speckle(h: int, w: int, rng: np.random.Generator, sigma: float, lo=40, hi=215) -> np.ndarray:
    tex = rng.uniform(lo, hi, (h, w)).astype(np.float32)
    return cv2.GaussianBlur(tex, (0, 0), sigma)


def build_canvas(scene: SceneSpec) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Static background + per-component sprites (with margin for sub-pixel warps)."""
    rng = np.random.default_rng(scene.seed + 1)
    h, w = scene.height, scene.width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    canvas = 70 + 25 * (xx / w) + 10 * (yy / h)
    canvas += _speckle(h, w, rng, 4.0, -8, 8)                      # low-texture painted wall
    cv2.rectangle(canvas, (4, 150), (w - 5, 182), 55, -1)          # machine bed (flat paint)
    cv2.line(canvas, (4, 150), (w - 5, 150), 120, 1)
    rx, ry, rw, rh = scene.reference_roi                           # static speckle target
    canvas[ry:ry + rh, rx:rx + rw] = _speckle(rh, rw, rng, 1.1)
    cv2.rectangle(canvas, (rx - 1, ry - 1), (rx + rw, ry + rh), 230, 1)

    sprites = {}
    for comp in scene.components:
        x, y, cw, ch = comp.region
        m = 8
        spr = np.full((ch + 2 * m, cw + 2 * m), 95, np.float32)
        # cast-metal body: low-frequency mottling + machined detail
        spr += _speckle(ch + 2 * m, cw + 2 * m, rng, 3.0, -8, 8)     # smooth painted casting
        # strong texture only on a sub-area (bolted cover / label); rest is smooth
        tx0, ty0 = m + cw // 6, m + ch // 5
        spr[ty0:ty0 + ch // 2, tx0:tx0 + (2 * cw) // 3] = _speckle(ch // 2, (2 * cw) // 3, rng, 1.0)
        for bx, by in [(m + 6, m + 6), (m + cw - 7, m + 6), (m + 6, m + ch - 7), (m + cw - 7, m + ch - 7)]:
            cv2.circle(spr, (bx, by), 3, 210, -1)
            cv2.circle(spr, (bx, by), 1, 40, -1)
        cv2.rectangle(spr, (m, m), (m + cw - 1, m + ch - 1), 200, 1)
        sprites[comp.name] = spr
    return canvas.astype(np.float32), sprites


class FourierShifter:
    """Exact band-limited sub-pixel translation (cv2.warpAffine quantises to 1/32 px
    and its cubic kernel is not an ideal shift, which would bias every tracker)."""

    def __init__(self, img: np.ndarray):
        self.F = np.fft.rfft2(img.astype(np.float64))
        h, w = img.shape
        self.shape = (h, w)
        self.ky = np.fft.fftfreq(h)[:, None]
        self.kx = np.fft.rfftfreq(w)[None, :]

    def shift(self, dx: float, dy: float) -> np.ndarray:
        ramp = np.exp(-2j * np.pi * (self.kx * dx + self.ky * dy))
        return np.fft.irfft2(self.F * ramp, s=self.shape).astype(np.float32)


def exposure_gain(scene: SceneSpec, n: int) -> np.ndarray:
    """Auto-exposure: the camera re-meters every few seconds and slews to a new
    gain (first-order lag ~0.3 s), plus slow drift. Separate RNG stream."""
    if scene.exposure_var <= 0:
        return np.ones(n)
    rng = np.random.default_rng(scene.seed + 17)
    fs = scene.fps
    target = np.ones(n)
    k = 0
    level = 1.0
    while k < n:
        k_next = k + int(rng.uniform(3.0, 12.0) * fs)
        level = 1.0 + scene.exposure_var * rng.uniform(-1, 1)
        target[k:k_next] = level
        k = k_next
    a = np.exp(-1.0 / (0.3 * fs))
    g = lfilter([1 - a], [1, -a], target, zi=[a * target[0]])[0]
    t = np.arange(n) / fs
    return g * (1.0 + 0.3 * scene.exposure_var * np.sin(2 * np.pi * t / 90.0))


def render_frames(scene: SceneSpec, truth: Dict[str, np.ndarray], shake: np.ndarray,
                  start: int = 0, stop: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield uint8 grey frames (like a real camera)."""
    canvas, sprites = build_canvas(scene)
    shifters = {k: FourierShifter(v) for k, v in sprites.items()}
    cv2.setRNGSeed(int(scene.seed) * 7919 + 2 + start)   # cv2.randn is ~5x faster than numpy
    noise = np.empty((scene.height, scene.width), np.float32)
    stop = scene.n_frames if stop is None else stop
    m = 8
    mean_level = float(canvas.mean())
    gain = exposure_gain(scene, scene.n_frames)
    rs = scene.rolling_shutter_s * scene.fps          # readout time in frames (top -> bottom)
    band = 6                                           # rows rendered with one displacement
    delayed = {}
    if rs > 0:
        # exact fractional delay of the (band-limited) true motion for every row band;
        # linear interpolation between frames would low-pass the motion (~50 % at 41 Hz)
        for comp in scene.components:
            x, y, cw, ch = comp.region
            tr = truth[comp.name]
            F = np.fft.rfft(tr, axis=1)
            fr_ = np.fft.rfftfreq(tr.shape[1])                     # cycles per frame
            for r0 in range(0, ch, band):
                tau = (y + r0 + band / 2) / scene.height * rs        # frames
                delayed[(comp.name, r0)] = np.fft.irfft(F * np.exp(2j * np.pi * fr_ * tau), tr.shape[1], axis=1)
    for k in range(start, stop):
        frame = canvas.copy()
        for comp in scene.components:
            x, y, cw, ch = comp.region
            tr = truth[comp.name]
            if rs <= 0:
                moved = shifters[comp.name].shift(tr[0, k], tr[1, k])
                frame[y:y + ch, x:x + cw] = moved[m:m + ch, m:m + cw]
            else:
                # rolling shutter: row r is exposed (y + r) / H * readout later than row 0
                for r0 in range(0, ch, band):
                    dd = delayed[(comp.name, r0)]
                    moved = shifters[comp.name].shift(dd[0, k], dd[1, k])
                    r1 = min(ch, r0 + band)
                    frame[y + r0:y + r1, x:x + cw] = moved[m + r0:m + r1, m:m + cw]
        if scene.shake_rms_px > 0:
            frame = FourierShifter(frame).shift(shake[0, k], shake[1, k])
        if scene.contrast != 1.0:
            frame = mean_level + scene.contrast * (frame - mean_level)
        if scene.flicker_amp > 0:
            frame *= 1.0 + scene.flicker_amp * np.sin(2 * np.pi * scene.flicker_hz * k / scene.fps)
        if scene.exposure_var > 0:
            frame *= gain[k]
        if scene.cam_noise > 0:
            cv2.randn(noise, 0.0, float(scene.cam_noise))
            frame += noise
        yield np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def h264_roundtrip(frames: Iterator[np.ndarray], fps: float, crf: int, workdir: str,
                   preset: str = "medium") -> Iterator[np.ndarray]:
    """Encode with libx264 (yuv420p, like a phone/NVR) and decode with OpenCV."""
    import os
    import subprocess
    import tempfile
    first = next(iter(frames))
    h, w = first.shape
    fd, path = tempfile.mkstemp(suffix=".mp4", dir=workdir)
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(first.tobytes())
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    size = os.path.getsize(path)
    cap = cv2.VideoCapture(path)
    h264_roundtrip.last_bytes = size
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    cap.release()
    os.remove(path)


def make_synthetic_run(scene: SceneSpec):
    """Convenience: truth + a frame iterator."""
    t, truth, shake = generate_displacements(scene)
    return t, truth, shake, render_frames(scene, truth, shake)
