"""Sub-pixel ROI displacement estimators built on OpenCV 5.

Three independent estimators so results can be cross-validated:

* ``lk``    - pyramidal Lucas-Kanade on Shi-Tomasi features inside the ROI
              (cv2.goodFeaturesToTrack + cv2.calcOpticalFlowPyrLK, forward-backward
              check, robust median). Always measured against the *reference*
              frame, so there is no drift accumulation.
* ``ic``    - inverse-compositional gradient estimator (cv2.GaussianBlur,
              cv2.Sobel, cv2.warpAffine) - dense sub-pixel tracker, default.
* ``ecc``   - Enhanced Correlation Coefficient (cv2.findTransformECC); invariant
              to illumination gain/offset, used for cross-validation.

All trackers take uint8 grey frames (what a camera delivers) and return
(dx, dy, quality in [0,1]).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]
MARGIN = 10


def crop(frame: np.ndarray, box: Box, margin: int = 0) -> np.ndarray:
    x, y, w, h = box
    H, W = frame.shape[:2]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
    return frame[y0:y1, x0:x1]


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def texture_score(gray: np.ndarray, box: Box) -> Dict[str, float]:
    """Trackability of a ROI: Shi-Tomasi min-eigenvalue statistics + gradient energy."""
    patch = crop(gray, box).astype(np.float32)
    if patch.size == 0:
        return {"min_eig_p50": 0.0, "min_eig_p90": 0.0, "grad_energy": 0.0, "n_features": 0}
    eig = cv2.cornerMinEigenVal(patch, blockSize=5, ksize=3)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    pts = cv2.goodFeaturesToTrack(patch.astype(np.uint8), maxCorners=200, qualityLevel=0.02, minDistance=3)
    return {
        "min_eig_p50": float(np.percentile(eig, 50)),   # texture spread over the ROI
        "min_eig_p90": float(np.percentile(eig, 90)),
        "grad_energy": float(np.mean(gx * gx + gy * gy)),
        "n_features": int(0 if pts is None else len(pts)),
    }


class LKTracker:
    name = "lk"

    def __init__(self, ref_gray: np.ndarray, box: Box, max_pts: int = 60, win: int = 11):
        self.box = box
        self.ref = crop(ref_gray, box, MARGIN).copy()
        x, y, w, h = box
        self.off = (min(MARGIN, x), min(MARGIN, y))
        mask = np.zeros_like(self.ref)
        ox, oy = self.off
        mask[oy:oy + h, ox:ox + w] = 255
        pts = cv2.goodFeaturesToTrack(self.ref, maxCorners=max_pts, qualityLevel=0.01,
                                      minDistance=3, blockSize=5, mask=mask)
        if pts is None or len(pts) < 4:      # fall back to a grid
            gx, gy = np.meshgrid(np.linspace(ox + 4, ox + w - 5, 6), np.linspace(oy + 4, oy + h - 5, 6))
            pts = np.stack([gx.ravel(), gy.ravel()], 1)[:, None, :]
        self.pts = pts.astype(np.float32)
        self.win = (win, win)
        self.crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-3)
        self.guess = self.pts.copy()

    def track(self, gray: np.ndarray) -> Tuple[float, float, float]:
        cur = crop(gray, self.box, MARGIN)
        if cur.shape != self.ref.shape:
            return np.nan, np.nan, 0.0
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.ref, cur, self.pts, self.guess.copy(), winSize=self.win,
                                              maxLevel=1, criteria=self.crit,
                                              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, self.ref, nxt, self.pts.copy(), winSize=self.win,
                                                maxLevel=1, criteria=self.crit,
                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        fb = np.linalg.norm((back - self.pts).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.1)
        if good.sum() < 3:
            return np.nan, np.nan, 0.0
        d = (nxt - self.pts).reshape(-1, 2)[good]
        dx, dy = np.median(d, axis=0)
        self.guess = self.pts + np.float32([dx, dy])
        quality = float(good.mean())
        return float(dx), float(dy), quality


class ECCTracker:
    """Enhanced Correlation Coefficient alignment (cv2.findTransformECC, translation model).

    ECC maximises a zero-mean normalised correlation, so it is invariant to
    global gain/offset of illumination - a useful independent check of the
    gradient tracker when lighting flickers.
    """
    name = "ecc"

    def __init__(self, ref_gray: np.ndarray, box: Box, iters: int = 40, eps: float = 1e-4, inset: int = 4):
        self.box = box
        x, y, w, h = box
        # inner template: keeps the template inside the search crop even when the
        # ROI touches the image border (the crop with margin is then truncated)
        inner = (x + inset, y + inset, w - 2 * inset, h - 2 * inset)
        self.tmpl = crop(ref_gray, inner).astype(np.float32)
        x0, y0 = max(0, x - MARGIN), max(0, y - MARGIN)
        self.off = np.array([inner[0] - x0, inner[1] - y0], np.float64)
        self.crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iters, eps)
        self.d = np.zeros(2)

    def track(self, gray: np.ndarray) -> Tuple[float, float, float]:
        cur = crop(gray, self.box, MARGIN).astype(np.float32)
        W = np.float32([[1, 0, self.off[0] + self.d[0]], [0, 1, self.off[1] + self.d[1]]])
        try:
            cc, W = cv2.findTransformECC(self.tmpl, cur, W, cv2.MOTION_TRANSLATION, self.crit, None, 1)
        except cv2.error:
            return np.nan, np.nan, 0.0
        self.d = np.array([W[0, 2], W[1, 2]], np.float64) - self.off
        return float(self.d[0]), float(self.d[1]), float(np.clip(cc, 0, 1))


class ICTracker:
    """Inverse-compositional translation estimator (fixes the compounding-warp bug of v1).

    photometric=True compares zero-mean, contrast-normalised patches (gain/offset
    from cv2.meanStdDev every iteration), so auto-exposure steps and lighting
    flicker do not turn into fake displacement."""
    name = "ic"

    def __init__(self, ref_gray: np.ndarray, box: Box, blur: float = 1.0, iters: int = 3,
                 photometric: bool = True):
        self.box, self.blur, self.iters, self.photometric = box, blur, iters, photometric
        ref = cv2.GaussianBlur(crop(ref_gray, box, MARGIN).astype(np.float32), (0, 0), blur)
        self.ref = ref
        x, y, w, h = box
        ox, oy = min(MARGIN, x), min(MARGIN, y)
        self.sl = (slice(oy, oy + h), slice(ox, ox + w))
        gx = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        self.gx, self.gy = gx[self.sl], gy[self.sl]
        H = np.array([[np.sum(self.gx * self.gx), np.sum(self.gx * self.gy)],
                      [np.sum(self.gx * self.gy), np.sum(self.gy * self.gy)]], np.float64)
        self.Hinv = np.linalg.inv(H + 1e-6 * np.eye(2))
        self.ref_c = ref[self.sl]
        m, sd = cv2.meanStdDev(np.ascontiguousarray(self.ref_c))
        self.ref_mu, self.ref_sd = float(m[0, 0]), max(float(sd[0, 0]), 1e-6)
        self.ref_z = self.ref_c - self.ref_mu
        self.d = np.zeros(2)

    def track(self, gray: np.ndarray) -> Tuple[float, float, float]:
        cur = cv2.GaussianBlur(crop(gray, self.box, MARGIN).astype(np.float32), (0, 0), self.blur)
        if cur.shape != self.ref.shape:
            return np.nan, np.nan, 0.0
        h, w = cur.shape
        d = self.d.copy()                                 # warm start from previous frame
        err = None
        for _ in range(self.iters):
            M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])  # sample cur at x + d (always from the original)
            warped = cv2.warpAffine(cur, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REFLECT101)
            wc = warped[self.sl]
            if self.photometric:
                m, sd = cv2.meanStdDev(np.ascontiguousarray(wc))
                err = (wc - float(m[0, 0])) * (self.ref_sd / max(float(sd[0, 0]), 1e-6)) - self.ref_z
            else:
                err = wc - self.ref_c
            b = np.array([np.sum(self.gx * err), np.sum(self.gy * err)])
            d -= self.Hinv @ b
        self.d = d
        resid = float(np.sqrt(np.mean(err * err))) if err is not None else 0.0
        quality = float(np.clip(1.0 - resid / 20.0, 0.0, 1.0))
        return float(d[0]), float(d[1]), quality


TRACKERS = {"ic": ICTracker, "lk": LKTracker, "ecc": ECCTracker}


def make_tracker(method: str, ref_gray: np.ndarray, box: Box):
    return TRACKERS[method](ref_gray, box)


def track_sequence(frames, boxes: Dict[str, Box], method: str = "ic",
                   ref_frame: Optional[np.ndarray] = None):
    """Track several ROIs over an iterable of frames.

    Returns ({name: (2, n) displacement}, {name: (n,) quality}, n_frames).
    """
    it = iter(frames)
    first = to_gray(next(it)) if ref_frame is None else to_gray(ref_frame)
    trackers = {k: make_tracker(method, first, b) for k, b in boxes.items()}
    disp = {k: [] for k in boxes}
    qual = {k: [] for k in boxes}
    if ref_frame is None:
        for k in boxes:
            disp[k].append((0.0, 0.0))
            qual[k].append(1.0)
    for fr in it:
        g = to_gray(fr)
        for k, tr in trackers.items():
            dx, dy, q = tr.track(g)
            disp[k].append((dx, dy))
            qual[k].append(q)
    out = {k: fill_nans(np.asarray(v, np.float64).T) for k, v in disp.items()}
    return out, {k: np.asarray(v) for k, v in qual.items()}, len(next(iter(qual.values())))


def fill_nans(xy: np.ndarray) -> np.ndarray:
    """Linear interpolation over dropped measurements."""
    xy = xy.copy()
    for a in range(xy.shape[0]):
        v = xy[a]
        bad = ~np.isfinite(v)
        if bad.all():
            xy[a] = 0.0
        elif bad.any():
            idx = np.arange(len(v))
            v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return xy
