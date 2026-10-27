"""Linear Eulerian motion magnification (Wu et al., SIGGRAPH 2012) with OpenCV pyramids.

Used as *evidence* for the human reviewer: sub-pixel vibration in a frequency
band is amplified until it is visible, and an x-t slice shows it at a glance.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt


def _down(img: np.ndarray, levels: int) -> np.ndarray:
    for _ in range(levels):
        img = cv2.pyrDown(img)
    return img


def _up(img: np.ndarray, levels: int, size: Tuple[int, int]) -> np.ndarray:
    for _ in range(levels):
        img = cv2.pyrUp(img)
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def magnify(frames: np.ndarray, fps: float, band: Tuple[float, float], amp: float = 25.0,
            levels: int = 1) -> np.ndarray:
    """frames: (n, h, w) uint8 -> magnified (n, h, w) uint8."""
    f32 = frames.astype(np.float32)
    small = np.stack([_down(f, levels) for f in f32])
    hi = min(band[1], 0.45 * fps)
    sos = butter(2, [band[0], hi], btype="bandpass", fs=fps, output="sos")
    filt = sosfiltfilt(sos, small, axis=0).astype(np.float32)
    h, w = frames.shape[1:]
    out = np.empty_like(f32)
    for i in range(len(frames)):
        out[i] = f32[i] + amp * _up(filt[i], levels, (w, h))
    return np.clip(out, 0, 255).astype(np.uint8)


def xt_slice(frames: np.ndarray, row: int, x0: int, x1: int) -> np.ndarray:
    """Space-time slice: one image row over time (time runs downwards)."""
    return frames[:, row, x0:x1]


def evidence_panel(frames: np.ndarray, fps: float, box, band, amp: float = 25.0,
                   seconds: float = 3.0, path: Optional[str] = None) -> np.ndarray:
    """Original vs magnified x-t slices through the ROI centre, side by side."""
    n = min(len(frames), int(seconds * fps))
    clip = frames[-n:]
    x, y, w, h = box
    pad = 12
    x0, x1 = max(0, x - pad), min(clip.shape[2], x + w + pad)
    row = y + h // 2
    mag = magnify(clip, fps, band, amp)
    a = xt_slice(clip, row, x0, x1)
    b = xt_slice(mag, row, x0, x1)
    scale = 4
    a = cv2.resize(a, (a.shape[1] * scale, a.shape[0]), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, (b.shape[1] * scale, b.shape[0]), interpolation=cv2.INTER_NEAREST)
    sep = np.full((a.shape[0], 6), 255, np.uint8)
    panel = cv2.cvtColor(np.hstack([a, sep, b]), cv2.COLOR_GRAY2BGR)
    if path:
        cv2.imwrite(path, panel)
    return panel


def write_video(frames: np.ndarray, fps: float, path: str, play_fps: float = 30.0) -> str:
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), play_fps, (w, h), isColor=False)
    for f in frames:
        vw.write(f)
    vw.release()
    return path


# --------------------------------------------------------------------------- Lagrangian
def lagrangian_magnify(crops: np.ndarray, disp: np.ndarray, fps: float, band: Tuple[float, float],
                       amp: float = 40.0) -> np.ndarray:
    """Re-warp each crop by (amp-1) x the *measured* band-passed displacement.

    Because the motion is already measured by the tracker, this avoids the
    intensity blobs that linear Eulerian magnification creates on texture."""
    hi = min(band[1], 0.45 * fps)
    sos = butter(2, [band[0], hi], btype="bandpass", fs=fps, output="sos")
    d = sosfiltfilt(sos, disp - disp.mean(axis=1, keepdims=True), axis=1) * (amp - 1.0)
    h, w = crops.shape[1:]
    out = np.empty_like(crops)
    for i in range(len(crops)):
        M = np.float32([[1, 0, d[0, i]], [0, 1, d[1, i]]])
        out[i] = cv2.warpAffine(crops[i], M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    return out


def comparison_panel(base_crops: np.ndarray, base_disp: np.ndarray, now_crops: np.ndarray,
                     now_disp: np.ndarray, fps: float, band: Tuple[float, float], amp: float = 40.0,
                     seconds: float = 1.5, path: Optional[str] = None) -> np.ndarray:
    """x-t slices (time downwards) of baseline vs now, both magnified identically."""
    n = int(seconds * fps)
    tiles = []
    for crops, disp, label in ((base_crops, base_disp, "baseline"), (now_crops, now_disp, "now")):
        m = lagrangian_magnify(crops[-n:], disp[:, -n:], fps, band, amp)
        row = m.shape[1] // 2
        sl = m[:, row, :]
        sl = cv2.resize(sl, (sl.shape[1] * 3, sl.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        sl = cv2.cvtColor(sl, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(sl, (0, 0), (sl.shape[1], 18), (255, 255, 255), -1)
        cv2.putText(sl, f"{label}  x{amp:.0f} {band[0]:.0f}-{band[1]:.0f}Hz", (4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(sl)
        tiles.append(np.full((sl.shape[0], 8, 3), 255, np.uint8))
    panel = np.hstack(tiles[:-1])
    if path:
        cv2.imwrite(path, panel)
    return panel
