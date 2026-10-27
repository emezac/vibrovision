"""Measurement theory: the image Jacobian fixes how well any camera can measure motion.

Brightness constancy, linearised around the reference template T:

    I_t(x) = T(x - u_t) + n_t(x)  ~  T(x) - J(x) u_t + n_t(x),     J(x) = grad T(x)   (1 x 2)

With white sensor noise n ~ N(0, sigma^2) the Fisher information of u is
F = J^T J / sigma^2, so for ANY unbiased estimator (Cramer-Rao)

    Cov(u_hat) >= sigma^2 (J^T J)^-1                                              (CRB)

* lambda_min(J^T J) is (up to the derivative filter) the Shi-Tomasi "min eigenvalue" used to pick ROIs, so ROI
  selection = maximising the worst-axis Fisher information. Not a heuristic.
* The inverse-compositional tracker solves the same normal equations on blurred images
  (H = sum g g^T, b = sum g (G * n)); a first-order white-noise covariance approximation is the sandwich

    Cov_IC = sigma^2 H^-1 [ sum_y (G * (g 1_ROI))(y) (G * (g 1_ROI))(y)^T ] H^-1      (IC)

  so efficiency compares this approximation with the known-template white-noise bound.
* In this linearized fixed-template approximation, reference noise adds a constant offset, so for f > 0 the
  tracker noise is white with one-sided PSD (summed over both axes)

    N0 = 2 tr(Cov_IC) / fps        [px^2 / Hz]                                      (floor)

  which is the noise floor subtracted before fitting alpha - predicted per ROI from one
  frame plus sigma, with no static reference target needed.

sigma is estimated from the video itself: the variance of temporal frame differences obeys
Var(I_t - I_{t-1})(x) = 2 sigma^2 + J(x) Cov(du) J(x)^T, a linear model in (1, gx^2, gy^2,
gx gy) across pixels whose intercept is 2 sigma^2 (motion is absorbed by the slopes).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .tracking import MARGIN, Box, crop, to_gray



def _sobel(img: np.ndarray):
    return (cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3) / 8.0,
            cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) / 8.0)


D5 = np.array([1.0, -8.0, 0.0, 8.0, -1.0], np.float32) / 12.0     # 4th-order central difference
D5_NOISE_GAIN = float(np.sum(D5.astype(np.float64) ** 2))           # Var(gx) = 0.903 sigma^2 for white noise


def fisher_matrix(gray: np.ndarray, box: Box, sigma: float = 0.0) -> np.ndarray:
    """J^T J of the (noise-free) template, debiased for the noise in the gradients of one frame.

    Uses a 4th-order central difference without cross-smoothing: Sobel's [1 2 1] smoothing underestimates the
    gradient energy of fine texture (and would make the "bound" exceed what the tracker achieves)."""
    patch = crop(to_gray(gray), box, 2).astype(np.float32)
    k = D5[::-1].copy()                                             # filter2D correlates
    gx = cv2.filter2D(patch, cv2.CV_32F, k.reshape(1, 5))
    gy = cv2.filter2D(patch, cv2.CV_32F, k.reshape(5, 1))
    x, y, w, h = box
    ox, oy = min(2, x), min(2, y)
    gx, gy = gx[oy:oy + h, ox:ox + w].astype(np.float64), gy[oy:oy + h, ox:ox + w].astype(np.float64)
    JtJ = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
    JtJ -= np.eye(2) * gx.size * D5_NOISE_GAIN * sigma ** 2
    return JtJ


def ic_noise_model(gray: np.ndarray, box: Box, sigma: float, fps: float, blur: float = 1.0) -> Dict[str, float]:
    """Predicted tracker covariance (sandwich), Cramer-Rao bound and the white noise floor."""
    g = to_gray(gray)
    JtJ = fisher_matrix(g, box, sigma)
    crb = sigma ** 2 * np.linalg.inv(JtJ + 1e-9 * np.eye(2))
    # exactly what ICTracker does: blur the crop-with-margin, Sobel, keep the ROI
    ref = cv2.GaussianBlur(crop(g, box, MARGIN).astype(np.float32), (0, 0), blur)
    x, y, w, h = box
    ox, oy = min(MARGIN, x), min(MARGIN, y)
    gx, gy = _sobel(ref)
    gx, gy = gx[oy:oy + h, ox:ox + w], gy[oy:oy + h, ox:ox + w]
    H = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]], np.float64)
    pad = int(np.ceil(4 * blur)) + 2
    S = np.zeros((2, 2))
    vs = []
    for gi in (gx, gy):
        z = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        z[pad:pad + h, pad:pad + w] = gi - gi.mean()          # photometric mean removal
        vs.append(cv2.GaussianBlur(z, (0, 0), blur, borderType=cv2.BORDER_CONSTANT).astype(np.float64))
    for i in range(2):
        for j in range(2):
            S[i, j] = np.sum(vs[i] * vs[j])
    Hi = np.linalg.inv(H)
    cov = sigma ** 2 * Hi @ S @ Hi
    ev = np.linalg.eigvalsh(JtJ)
    return {
        "sigma": float(sigma),
        "fisher_lambda_min": float(ev[0]),
        "fisher_lambda_max": float(ev[1]),
        "crb_rms_px": float(np.sqrt(np.trace(crb))),
        "ic_rms_px": float(np.sqrt(np.trace(cov))),
        "efficiency": float(np.trace(crb) / np.trace(cov)),
        "floor_psd": float(2.0 * np.trace(cov) / fps),
        "worst_axis_crb_px": float(sigma / np.sqrt(max(ev[0], 1e-12))),
        "scope": "first-order white-noise approximation; codec and model error excluded",
    }


def nuisance_fisher(J, nuisance, covariance=None):
    """Information after eliminating gain/offset by a Schur complement.

    J: pixels x motion parameters. nuisance: pixels x nuisance parameters.
    covariance must be positive definite; omitted means unit white variance.
    """
    J, Z = np.asarray(J, float), np.asarray(nuisance, float)
    if J.ndim != 2 or Z.ndim != 2 or len(J) != len(Z):
        raise ValueError("J and nuisance need matching pixel rows")
    if covariance is None:
        WJ, WZ = J, Z
    else:
        covariance = np.asarray(covariance, float)
        np.linalg.cholesky(covariance)
        WJ, WZ = np.linalg.solve(covariance, J), np.linalg.solve(covariance, Z)
    full = J.T @ WJ
    cross = J.T @ WZ
    effective = full - cross @ np.linalg.pinv(Z.T @ WZ) @ cross.T
    return (effective + effective.T) / 2


def estimate_sensor_noise(frames: Sequence[np.ndarray], box: Optional[Box] = None) -> Dict[str, float]:
    """sigma from temporal differences, regressing out motion through the image gradients.

    Var_t(I_t - I_{t-1})(x) = 2 sigma^2 + a gx^2 + b gy^2 + c gx gy  (least squares over pixels).
    Pixels in the top 2 % of gradient magnitude are dropped (second-order terms dominate there)."""
    st = np.stack([crop(to_gray(f), box).astype(np.float32) if box else to_gray(f).astype(np.float32)
                   for f in frames])
    d = np.diff(st, axis=0)
    v = d.var(axis=0).ravel().astype(np.float64)
    gx, gy = _sobel(st.mean(axis=0))
    gx, gy = gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)
    keep = (gx * gx + gy * gy) <= np.percentile(gx * gx + gy * gy, 98)
    A = np.stack([np.ones_like(v), gx * gx, gy * gy, gx * gy], 1)[keep]
    coef, *_ = np.linalg.lstsq(A, v[keep], rcond=None)
    # robust second pass (drop pixels with large residuals: occlusions, saturation)
    r = v[keep] - A @ coef
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    ok = np.abs(r) < 4 * s
    coef, *_ = np.linalg.lstsq(A[ok], v[keep][ok], rcond=None)
    sig2 = max(coef[0] / 2.0, 0.0)
    return {"sigma": float(np.sqrt(sig2)), "n_pixels": int(ok.sum()), "n_frames": int(len(st))}
