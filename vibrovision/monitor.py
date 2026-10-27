"""Streaming multi-ROI vibrometry engine.

Keeps (a) a continuous displacement history per ROI and (b) a ring buffer of
recent frames. The ring buffer is what lets the agent *act* on the video:
relocating a ROI or cross-checking with another estimator re-runs OpenCV on
the buffered frames instead of merely annotating a fixed result.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .spectral import SpectralConfig, noise_floor_from_reference
from .tracking import Box, crop, fill_nans, make_tracker, texture_score, to_gray

REF = "__reference__"


class StreamingMonitor:
    def __init__(self, fps: float, boxes: Dict[str, Box], reference_box: Optional[Box],
                 method: str = "ic", buffer_s: float = 20.0):
        self.fps = float(fps)
        self.method = method
        self.boxes: Dict[str, Box] = dict(boxes)
        if reference_box is not None:
            self.boxes[REF] = reference_box
        self.buffer_n = int(buffer_s * fps)
        self.buf: Optional[np.ndarray] = None
        self.n = 0                                   # frames seen
        self.trackers = {}
        self.disp: Dict[str, list] = {}
        self.qual: Dict[str, list] = {}
        self.start: Dict[str, int] = {}              # frame index where each ROI's history starts
        self.energy: Dict[str, float] = {}
        self.compensate = False
        self.track_seconds = 0.0

    # ------------------------------------------------------------------ ingest
    def push(self, frame: np.ndarray) -> None:
        g = to_gray(frame)
        if self.buf is None:
            self.buf = np.zeros((self.buffer_n,) + g.shape, np.uint8)
            self.first = g.copy()
            for name, box in self.boxes.items():
                self._start_roi(name, box, g, self.n)
        self.buf[self.n % self.buffer_n] = g
        t0 = time.perf_counter()
        for name, tr in self.trackers.items():
            dx, dy, q = tr.track(g)
            self.disp[name].append((dx, dy))
            self.qual[name].append(q)
        self.track_seconds += time.perf_counter() - t0
        self.n += 1

    def _start_roi(self, name: str, box: Box, ref: np.ndarray, at: int) -> None:
        self.trackers[name] = make_tracker(self.method, ref, box)
        self.boxes[name] = box
        self.disp[name], self.qual[name] = [], []
        self.start[name] = at
        ts = texture_score(ref, box)
        self.energy[name] = ts["grad_energy"] * box[2] * box[3]

    # ------------------------------------------------------------------ queries
    @property
    def t(self) -> float:
        return self.n / self.fps

    def buffered(self, seconds: float) -> np.ndarray:
        """Most recent frames (oldest first), at most the ring-buffer length."""
        k = min(max(1, int(round(seconds * self.fps))), self.n, self.buffer_n)
        idx = [(self.n - k + i) % self.buffer_n for i in range(k)]
        return self.buf[idx]

    def available_s(self, name: str) -> float:
        return len(self.disp[name]) / self.fps

    def window(self, name: str, seconds: float, compensate: Optional[bool] = None,
               end: Optional[int] = None) -> np.ndarray:
        """Displacement of `name` over `seconds` ending at absolute frame `end`
        (default: now). With compensation the static-target motion recorded on
        the *same* frames is subtracted."""
        comp = self.compensate if compensate is None else compensate
        end = self.n if end is None else end
        k = int(round(seconds * self.fps))
        s0 = self.start[name]
        lo, hi = max(0, end - k - s0), max(0, end - s0)
        xy = fill_nans(np.asarray(self.disp[name][lo:hi], float).reshape(-1, 2).T)
        if comp and REF in self.disp and name != REF and xy.shape[1]:
            r0 = self.start[REF]
            ref = np.asarray(self.disp[REF][lo + s0 - r0:hi + s0 - r0], float).reshape(-1, 2).T
            xy = xy - fill_nans(ref)
        return xy

    def quality(self, name: str, seconds: float) -> float:
        k = int(round(seconds * self.fps))
        q = np.asarray(self.qual[name][-k:], float)
        return float(np.nanmean(q)) if q.size else 0.0

    def noise_floor(self, name: str, seconds: float, cfg: SpectralConfig,
                    compensate: Optional[bool] = None) -> float:
        """White tracker-noise PSD for ROI `name`, scaled from the static reference
        by texture energy (noise variance ~ sigma^2 / sum |grad I|^2)."""
        if REF not in self.disp:
            return 0.0
        n_ref = noise_floor_from_reference(self.window(REF, seconds, compensate=False), cfg)
        scale = self.energy[REF] / max(self.energy[name], 1e-9)
        n_roi = n_ref * scale
        comp = self.compensate if compensate is None else compensate
        return n_roi + n_ref if comp else n_roi

    def reference_motion(self, seconds: float) -> Dict[str, float]:
        xy = self.window(REF, seconds, compensate=False)
        xy = xy - xy.mean(axis=1, keepdims=True)
        return {"rms_px": float(np.sqrt(np.sum(xy.var(axis=1)))),
                "peak_px": float(np.max(np.abs(xy))) if xy.size else 0.0}

    # ------------------------------------------------------------------ actions
    def best_box_in(self, region: Box, size: Tuple[int, int], step: int = 3) -> Tuple[Box, float]:
        """Most trackable sub-window of `region` (Shi-Tomasi min-eigenvalue map,
        integral image for O(1) box sums)."""
        g = self.buffered(1.0 / self.fps)[-1]
        x, y, w, h = region
        bw, bh = size
        patch = g[y:y + h, x:x + w].astype(np.float32)
        eig = cv2.cornerMinEigenVal(patch, blockSize=5, ksize=3)
        # score = median-like robust texture: sum of min(eig, cap) so a single bolt
        # or edge cannot dominate (the cap is the 75th percentile of the region)
        cap = float(np.percentile(eig, 75))
        integ = cv2.integral(np.minimum(eig, cap))
        best, best_s = None, -1.0
        for yy in range(2, h - bh - 1, step):
            for xx in range(2, w - bw - 1, step):
                s = integ[yy + bh, xx + bw] - integ[yy, xx + bw] - integ[yy + bh, xx] + integ[yy, xx]
                if s > best_s:
                    best_s, best = s, (x + xx, y + yy, bw, bh)
        if best is None:                    # region no larger than the ROI: nothing to search
            best = (x, y, min(bw, w), min(bh, h))
            best_s = float(np.minimum(eig, cap).mean()) * bw * bh
        return best, float(best_s / (bw * bh))

    def relocate(self, name: str, new_box: Box) -> None:
        """Move a ROI and back-fill its history from the frame buffer."""
        frames = self.buffered(self.buffer_n / self.fps)
        self._start_roi(name, new_box, frames[0], self.n - len(frames))
        tr = self.trackers[name]
        for fr in frames:
            dx, dy, q = tr.track(fr)
            self.disp[name].append((dx, dy))
            self.qual[name].append(q)

    def retrack(self, name: str, seconds: float, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Independent re-measurement of the last `seconds` with another estimator
        (and of the reference target, so compensation stays consistent)."""
        frames = self.buffered(seconds)
        out = {}
        for nm in ([name, REF] if REF in self.boxes else [name]):
            tr = make_tracker(method, frames[0], self.boxes[nm])
            d = [(0.0, 0.0)] + [tr.track(fr)[:2] for fr in frames[1:]]
            out[nm] = fill_nans(np.asarray(d, float).T)
        xy = out[name]
        if self.compensate and REF in out:
            xy = xy - out[REF]
        return xy, out.get(REF)
