"""Joint spectral evidence; descriptive comparisons, not universal fault labels."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigvalsh
from scipy.signal import csd


def cross_spectrum(signals, fs, nperseg=None):
    """Hermitian spectral matrix: (frequency, channel, channel)."""
    x = np.asarray(signals, float)
    if x.ndim != 2 or x.shape[1] < 32 or not np.isfinite(x).all():
        raise ValueError("requires finite (channels, samples) with >=32 samples")
    nper = min(nperseg or max(32, 2 ** int(np.floor(np.log2(2 * fs)))), x.shape[1])
    matrices = None
    for i in range(len(x)):
        for j in range(i, len(x)):
            f, p = csd(x[i], x[j], fs=fs, nperseg=nper, noverlap=nper // 2, detrend="linear")
            if matrices is None:
                matrices = np.empty((len(f), len(x), len(x)), complex)
            matrices[:, i, j], matrices[:, j, i] = p, np.conj(p)
    return f, matrices


def relative_spectral_evidence(base, current, fs_base, fs_current=None):
    """Compare generalized eigenvalues over a shared frequency grid.

    Diagonal-relative shrinkage is equivariant to fixed diagonal channel gains.
    Results are descriptive: no calibrated detection threshold or causal label.
    """
    f0, S0 = cross_spectrum(base, fs_base)
    ft, St = cross_spectrum(current, fs_current or fs_base)
    if S0.shape[1:] != St.shape[1:]:
        raise ValueError("baseline and current must have the same channels")
    keep = (f0 >= 1.0) & (f0 < min(fs_base, fs_current or fs_base) * 0.45)
    grid, S0 = f0[keep], S0[keep]
    target = np.empty_like(S0)
    for i in range(S0.shape[1]):
        for j in range(S0.shape[2]):
            target[:, i, j] = np.interp(grid, ft, St[:, i, j].real) + 1j * np.interp(grid, ft, St[:, i, j].imag)
    values = []
    for A, B in zip(S0, target):
        # Joint scale-aware floor; scalar time-varying gain profiling below is approximate with this floor.
        A = 0.98 * A + 0.02 * np.diag(np.diag(A).real)
        B = 0.98 * B + 0.02 * np.diag(np.diag(B).real)
        scale = np.maximum(np.diag(A).real, 1e-30)
        A = A + np.diag(1e-8 * scale)
        B = B + np.diag(1e-8 * scale)
        try:
            values.append(eigvalsh(B, A))
        except np.linalg.LinAlgError:
            values.append(np.full(len(A), np.nan))
    values = np.asarray(values)
    good = np.isfinite(values).all(1) & (values > 0).all(1)
    if not good.any():
        return {"available": False, "reason": "spectral matrices could not be resolved"}
    logs = np.log(values[good])
    offset = float(np.mean(logs))
    centered = logs - offset
    per_frequency = np.sqrt(np.mean(centered ** 2, axis=1))
    return {"available": True, "shape_distance": float(np.sqrt(np.mean(centered ** 2))),
            "log_power_gain": offset, "frequency_hz": grid[good].tolist(),
            "generalized_eigenvalues": values[good].tolist(),
            "most_changed_band_hz": float(grid[good][np.argmax(per_frequency)]),
            "scope": "descriptive shared-coordinate spectral comparison; not a calibrated fault probability or stiffness estimate",
            "regularization": "2% diagonal-relative shrinkage and 1e-8 baseline diagonal floor"}
