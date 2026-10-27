"""Fast surrogate of the camera measurement chain, for long-horizon statistics.

Rendering + tracking runs ~4x real time on a laptop core, so hours of healthy
monitoring (needed to estimate false-alarm rates of ~1/day) are impractical
end-to-end. The surrogate replaces "render -> OpenCV tracker" by
"true displacement + Gaussian tracker error with the *measured* error spectrum",
where the error spectrum is calibrated from an end-to-end run
(`calibrate_error_model`). It is only trusted after it reproduces the
end-to-end indicator statistics on the same seeds (see eval/run_longrun.py).
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import numpy as np

from .monitor import REF
from .spectral import SpectralConfig, psd


def calibrate_error_model(scene, seconds: float = 60.0) -> dict:
    """Measure tracker error PSD (moving parts and static target) end-to-end."""
    from dataclasses import replace
    from .pipeline import track_scene
    sc = replace(scene, duration_s=seconds)
    t, truth, shake, disp, energy = track_scene(sc)
    cfg = SpectralConfig(fs=sc.fps)
    moving = []
    for k, d in disp.items():
        if k == REF:
            continue
        tr = truth[k] - truth[k][:, :1]
        f, P = psd(d - tr, cfg)
        moving.append(P)
    f, Pref = psd(disp[REF], cfg)
    return {"f": f.tolist(), "P_moving": np.mean(moving, axis=0).tolist(), "P_ref": Pref.tolist(),
            "fps": sc.fps, "cam_noise": sc.cam_noise}


def colored_error(n: int, fs: float, f_model: np.ndarray, P_model: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """2-axis Gaussian noise whose summed one-sided PSD follows (f_model, P_model)."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    target = np.interp(f, f_model, P_model) / 2.0          # per axis
    out = np.empty((2, n))
    for a in range(2):
        X = np.fft.rfft(rng.standard_normal(n))
        x = np.fft.irfft(X * np.sqrt(target * fs / 2.0), n)  # white(1) has one-sided PSD 2/fs
        out[a] = x
    return out


def surrogate_measurements(truth: Dict[str, np.ndarray], model: dict, rng: np.random.Generator,
                           shake: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Displacements as the tracker would report them (+ the static reference)."""
    fs = model["fps"]
    f, Pm, Pr = (np.asarray(model[k]) for k in ("f", "P_moving", "P_ref"))
    n = next(iter(truth.values())).shape[1]
    sh = np.zeros((2, n)) if shake is None else shake
    out = {k: v - v[:, :1] + sh + colored_error(n, fs, f, Pm, rng) for k, v in truth.items()}
    out[REF] = sh + colored_error(n, fs, f, Pr, rng)
    return out


def save_model(model: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(model, fh)


def load_model(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)
