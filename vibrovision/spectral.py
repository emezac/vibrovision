"""Spectral health indicators for camera-measured displacement.

Primary indicator: the broadband spectral exponent alpha in S(f) ~ f^-alpha,
fitted by weighted least squares (95 % CI; Theil-Sen optional) on a *noise-floor-corrected*, *tone-free*,
*log-binned* PSD. Rationale:

* tone-free   - rotation harmonics are expected and carry no wear information
                on their own; they are detected automatically (no a-priori
                fault frequency is given to the estimator).
* noise floor - camera/tracker noise is white and flattens the spectrum,
                which would mimic a fault. The floor is estimated from a static
                reference target and scaled by each ROI's texture strength.
* log bins    - equal weight per octave; otherwise the many high-frequency
                bins dominate the log-log regression.
* gain-free   - alpha is a slope in log-log space, so it does not change when
                the camera moves closer/further (unknown px -> mm scale),
                unlike RMS or band energy.

Baselines computed on exactly the same windows for a fair comparison:
RMS, high-frequency band RMS, high/low band power ratio, spectral flatness,
and the kurtosis of the high-passed signal (classic impact indicator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis, theilslopes
from scipy.stats import t as t_dist

INDICATORS = ("alpha", "band_ratio", "flatness", "kurtosis", "hf_rms", "rms")
# direction in which each indicator moves when a fault develops
DIRECTION = {"alpha": "down", "band_ratio": "up", "flatness": "up", "kurtosis": "up",
             "hf_rms": "up", "rms": "up"}


@dataclass
class SpectralConfig:
    fs: float = 120.0
    nperseg: int = 256
    fit_band: Tuple[float, float] = (2.0, 55.0)
    lo_band: Tuple[float, float] = (2.0, 10.0)
    hi_band: Tuple[float, float] = (25.0, 55.0)
    n_log_bins: int = 18
    tone_db: float = 6.0          # bins this far above the local median are tonal
    tone_kernel: int = 11
    min_snr: float = 2.0          # keep bins with P > min_snr * noise floor
    min_points: int = 8           # log-bins needed for a fit
    fit: str = "wls"              # "wls": weighted LS (weights = #bins per log-bin); "theil": Theil-Sen

    @classmethod
    def for_fps(cls, fps: float, **kw) -> "SpectralConfig":
        """Bands scaled to the camera's Nyquist frequency. At 120 fps this returns
        exactly the defaults used in all evaluations; a 30 fps phone clip gets
        fit 1.0-13.7 Hz, high band 6.3-13.7 Hz, ~2 s Welch segments."""
        if abs(fps - 120.0) < 1e-6:
            return cls(fs=fps, **kw)
        nyq = fps / 2.0
        nper = int(2 ** round(np.log2(fps * 256 / 120.0)))
        f_lo = min(2.0, max(0.5, 4.0 * fps / nper))           # >= 4 frequency bins above DC
        base = dict(fs=fps, nperseg=nper, fit_band=(f_lo, 0.917 * nyq), lo_band=(f_lo, max(0.167 * nyq, 2.5 * f_lo)),
                    hi_band=(0.417 * nyq, 0.917 * nyq),
                    n_log_bins=18 if fps >= 100 else 10, min_points=8 if fps >= 100 else 6)
        base.update(kw)
        return cls(**base)


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least squares line; the variance of a log-bin mean falls with the
    number of FFT bins it averages. Returns slope, intercept, slope standard error
    (from weighted residuals, so tone leakage or curvature widens the CI)."""
    W = w / w.sum()
    xm, ym = np.sum(W * x), np.sum(W * y)
    sxx = np.sum(W * (x - xm) ** 2)
    slope = np.sum(W * (x - xm) * (y - ym)) / sxx
    icpt = ym - slope * xm
    res = y - (icpt + slope * x)
    dof = max(len(x) - 2, 1)
    s2 = np.sum(w * res ** 2) / dof
    se = np.sqrt(s2 / np.sum(w * (x - xm) ** 2))
    return float(slope), float(icpt), float(se)


def psd(xy: np.ndarray, cfg: SpectralConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD summed over axes (trace of the cross-spectral matrix: rotation invariant)."""
    xy = np.atleast_2d(xy)
    nper = min(cfg.nperseg, xy.shape[1])
    f, P = welch(xy, fs=cfg.fs, nperseg=nper, noverlap=nper // 2, window="hann", detrend="linear", axis=-1)
    return f, P.sum(axis=0)


def tonal_mask(f: np.ndarray, P: np.ndarray, cfg: SpectralConfig) -> np.ndarray:
    logp = np.log10(np.maximum(P, 1e-20))
    bg = median_filter(logp, size=cfg.tone_kernel, mode="nearest")
    tone = (logp - bg) > cfg.tone_db / 10.0
    tone[0] = False
    grown = tone.copy()                     # grow by one bin each side (Hann leakage)
    grown[1:] |= tone[:-1]
    grown[:-1] |= tone[1:]
    return grown


def find_tones(f: np.ndarray, P: np.ndarray, cfg: SpectralConfig, top: int = 4) -> List[Tuple[float, float]]:
    logp = np.log10(np.maximum(P, 1e-20))
    bg = median_filter(logp, size=cfg.tone_kernel, mode="nearest")
    prom = 10 * (logp - bg)
    peaks = [(float(f[i]), float(prom[i])) for i in range(1, len(f) - 1)
             if prom[i] > cfg.tone_db and prom[i] >= prom[i - 1] and prom[i] >= prom[i + 1]]
    return sorted(peaks, key=lambda p: -p[1])[:top]


def _band_power(f, Pc, valid, band):
    m = valid & (f >= band[0]) & (f <= band[1])
    if m.sum() < 2:
        return np.nan
    df = f[1] - f[0]
    # interpolate over excluded bins so tone removal does not bias band energy
    sel = (f >= band[0]) & (f <= band[1])
    vals = np.interp(f[sel], f[m], Pc[m])
    return float(vals.sum() * df)


def analyze_window(xy: np.ndarray, cfg: SpectralConfig, noise_psd: float = 0.0) -> Dict[str, float]:
    """All indicators for one window of 2-axis displacement (px)."""
    xy = np.atleast_2d(xy)
    xy = xy - xy.mean(axis=1, keepdims=True)
    f, P = psd(xy, cfg)
    tones = tonal_mask(f, P, cfg)
    band = (f >= cfg.fit_band[0]) & (f <= cfg.fit_band[1])
    snr_ok = P > cfg.min_snr * max(noise_psd, 1e-20)
    valid = band & ~tones & snr_ok
    Pc = np.maximum(P - noise_psd, 1e-20)

    out = {"alpha": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "ci_width": np.nan,
           "n_points": 0, "valid_frac": float(valid.sum() / max(band.sum(), 1))}
    # log-binned PSD
    edges = np.logspace(np.log10(cfg.fit_band[0]), np.log10(cfg.fit_band[1]), cfg.n_log_bins + 1)
    lf, lp, wt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = valid & (f >= lo) & (f < hi)
        if m.any():
            lf.append(np.log10(np.sqrt(lo * hi)))
            lp.append(np.log10(Pc[m].mean()))
            wt.append(m.sum())
    if len(lf) >= cfg.min_points:
        lf, lp, wt = map(np.asarray, (lf, lp, wt))
        if cfg.fit == "theil":
            slope, icpt, lo_s, hi_s = theilslopes(lp, lf, alpha=0.95)
        else:
            slope, icpt, se = _wls(lf, lp, wt.astype(float))
            q = t_dist.ppf(0.975, len(lf) - 2)
            lo_s, hi_s = slope - q * se, slope + q * se
        out.update(alpha=float(-slope), ci_lo=float(-hi_s), ci_hi=float(-lo_s),
                   ci_width=float(hi_s - lo_s), n_points=len(lf), intercept=float(icpt))

    p_lo = _band_power(f, Pc, valid, cfg.lo_band)
    p_hi = _band_power(f, Pc, valid, cfg.hi_band)
    out["band_ratio"] = float(np.log10(p_hi / p_lo)) if (p_lo and p_hi and p_lo > 0 and p_hi > 0) else np.nan
    out["hf_rms"] = float(np.sqrt(p_hi)) if p_hi == p_hi else np.nan
    v = Pc[valid]
    out["flatness"] = float(np.exp(np.mean(np.log(v))) / np.mean(v)) if v.size > 4 else np.nan
    out["rms"] = float(np.sqrt(np.sum(xy.var(axis=1))))
    sos = butter(4, cfg.hi_band[0], btype="highpass", fs=cfg.fs, output="sos")
    hp = sosfiltfilt(sos, xy, axis=1)
    out["kurtosis"] = float(np.mean(kurtosis(hp, axis=1, fisher=False)))
    # tonal prominence: power in detected tones vs broadband power in the fit band (gain-invariant).
    # alpha ignores tones on purpose (impacts are broadband); an imbalance is a tone at 1X.
    tb = band & tones
    bb = band & ~tones
    out["tonal_db"] = float(10 * np.log10(max(Pc[tb].sum(), 1e-20) / max(Pc[bb].sum(), 1e-20))) if tb.any() else -30.0
    tl = find_tones(f, P, cfg, top=1)
    out["tone_hz"] = tl[0][0] if tl else np.nan
    total = float(P[band].sum())
    out["snr_db"] = float(10 * np.log10(total / max(noise_psd * band.sum(), 1e-20))) if noise_psd > 0 else np.inf
    return out


def alpha_mixture(healthy_psd, added_psd, healthy_alpha, added_alpha, severity=1.0):
    """Exact local slope for a sum of two power laws, not a universal damage law."""
    h, d = np.asarray(healthy_psd, float), np.asarray(added_psd, float) * severity
    if np.any(h <= 0) or np.any(d < 0) or severity < 0:
        raise ValueError("positive healthy PSD and nonnegative added PSD required")
    return (healthy_alpha * h + added_alpha * d) / (h + d)


def spectrum_for_plot(xy: np.ndarray, cfg: SpectralConfig, noise_psd: float = 0.0):
    xy = np.atleast_2d(xy)
    f, P = psd(xy - xy.mean(axis=1, keepdims=True), cfg)
    return f, P, tonal_mask(f, P, cfg), noise_psd


def excess_band(f: np.ndarray, P: np.ndarray, P_base: np.ndarray, cfg: SpectralConfig,
                half_width: float = 8.0) -> Tuple[float, Tuple[float, float]]:
    """Where did the broadband spectrum grow most relative to baseline? That is
    where impacts ring, hence the band to demodulate."""
    from scipy.ndimage import uniform_filter1d
    tm = tonal_mask(f, P, cfg) | tonal_mask(f, P_base, cfg)
    r = uniform_filter1d(np.log10(np.maximum(P, 1e-20) / np.maximum(P_base, 1e-20)), 7)
    r[tm | (f < cfg.fit_band[0]) | (f > cfg.fit_band[1])] = -np.inf
    fp = float(f[int(np.argmax(r))])
    return fp, (max(cfg.fit_band[0], fp - half_width), min(0.48 * cfg.fs, fp + half_width))


def envelope_spectrum(xy: np.ndarray, cfg: SpectralConfig, band: Optional[Tuple[float, float]] = None,
                      search: Tuple[float, float] = (2.0, 20.0)) -> Dict[str, float]:
    """Classic bearing diagnosis: demodulate the band where impacts ring and look
    for the repetition rate in the spectrum of the envelope.

    Demodulation is done in the Fourier domain with rotation tones notched out
    first - otherwise a shaft harmonic inside the band beats with the impact
    comb and produces a spurious envelope line."""
    xy = np.atleast_2d(xy)
    xy = xy - xy.mean(axis=1, keepdims=True)
    band = band or cfg.hi_band
    n = xy.shape[1]
    fr = np.fft.fftfreq(n, 1.0 / cfg.fs)
    f_psd, P = psd(xy, cfg)
    tones = [ft for ft, _ in find_tones(f_psd, P, cfg, top=8)]
    keep = (fr >= band[0]) & (fr <= band[1])
    df_notch = max(0.35, 1.5 * cfg.fs / cfg.nperseg)
    for ft in tones:
        keep &= np.abs(fr - ft) > df_notch
    win = np.hanning(n)
    env = np.zeros(n)
    for ax in range(xy.shape[0]):
        X = np.fft.fft(xy[ax] * win)
        env += np.abs(np.fft.ifft(2 * X * keep))
    env = (env - env.mean())[n // 10: -n // 10]          # drop window edges
    nper = min(len(env), 2 * cfg.nperseg)
    f, Pe = welch(env, fs=cfg.fs, nperseg=nper, noverlap=nper // 2)
    # line detection: normalise by the local median, which removes the broad
    # low-frequency hump that the envelope of band-limited noise always has
    L = 10 * np.log10(np.maximum(Pe, 1e-30) / np.maximum(median_filter(Pe, size=15, mode="nearest"), 1e-30))
    m = (f >= search[0]) & (f <= search[1])
    if not m.any():
        return {"peak_hz": np.nan, "prominence_db": 0.0}
    fm, Lm = f[m], L[m]
    i = int(np.argmax(Lm))
    # sub-harmonic check: an impact comb also has lines at 2f, 3f...; if a
    # comparable line exists at f/2, the repetition rate is f/2
    while True:
        target = fm[i] / 2.0
        if target < search[0]:
            break
        j = np.where(np.abs(fm - target) <= 2.5 * (f[1] - f[0]))[0]
        if j.size == 0:
            break
        j = j[np.argmax(Lm[j])]
        if Lm[j] >= max(4.0, Lm[i] - 6.0):
            i = int(j)
        else:
            break
    return {"peak_hz": float(fm[i]), "prominence_db": float(Lm[i]), "L_db": L.tolist(), "f_all": f.tolist(),
            "band_hz": [float(band[0]), float(band[1])], "notched_tones_hz": [round(t, 2) for t in tones],
            "f": f[m].tolist(), "P": Pe[m].tolist()}


def tone_power_db(xy: np.ndarray, cfg: SpectralConfig, tones_hz: Sequence[float], half_width: float = 0.75) -> float:
    """Displacement power (dB re 1 px^2) within +/- half_width Hz of known rotation harmonics.

    A forcing change (imbalance, misalignment) grows these lines; unlike tonal_db it does not depend on
    whether the tone detector fires in a given window."""
    f, P = psd(np.atleast_2d(xy) - np.atleast_2d(xy).mean(axis=1, keepdims=True), cfg)
    df = f[1] - f[0]
    m = np.zeros_like(f, bool)
    for t in tones_hz:
        m |= np.abs(f - t) <= max(half_width, df)
    return float(10 * np.log10(max(P[m].sum() * df, 1e-20)))


def noise_floor_from_reference(ref_xy: np.ndarray, cfg: SpectralConfig, min_hz: float = 15.0) -> float:
    """White tracker-noise PSD level from a static target (median above min_hz
    so slow camera sway does not inflate it)."""
    f, P = psd(np.atleast_2d(ref_xy), cfg)
    lower = min(min_hz, 0.75 * cfg.fit_band[1])
    m = (f >= lower) & (f <= cfg.fit_band[1])
    return float(np.median(P[m])) if m.any() else 0.0


def defect_line_scores(env: Dict, candidates: Dict[str, float], harmonics: int = 2) -> Dict[str, float]:
    """Hypothesis test with known bearing kinematics: summed line prominence (dB
    above local median) at each candidate defect frequency and its harmonics."""
    f = np.asarray(env.get("f_all", []))
    L = np.asarray(env.get("L_db", []))
    if f.size < 3:
        return {}
    df = f[1] - f[0]
    out = {}
    for name, f0 in candidates.items():
        sc = 0.0
        for h in range(1, harmonics + 1):
            m = np.abs(f - h * f0) <= max(2.0 * df, 0.02 * h * f0)
            sc += float(L[m].max()) if m.any() else 0.0
        out[name] = sc
    return out
