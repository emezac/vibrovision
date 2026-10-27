"""Sequential change detection on health indicators (one-sided CUSUM)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# minimum baseline spread per indicator (avoids absurd z-scores when the
# calibration windows happen to agree very closely)
SD_FLOOR = {"alpha": 0.04, "band_ratio": 0.03, "flatness": 0.01, "kurtosis": 0.05,
            "hf_rms": 1e-4, "rms": 1e-3}


@dataclass
class Cusum:
    direction: str = "down"          # "down": alarm when the indicator decreases
    k: float = 0.5                   # allowance (in baseline sd)
    h: float = 6.0                   # decision threshold
    sd_floor: float = 0.0
    weight: float = 1.0              # step/window: overlapping windows carry less new information
    mu: float = np.nan
    sd: float = np.nan
    g: float = 0.0
    history: List[float] = field(default_factory=list)

    def calibrate(self, values: Sequence[float]) -> None:
        v = np.asarray([x for x in values if np.isfinite(x)], float)
        if v.size == 0:
            raise ValueError("no finite calibration values")
        self.mu = float(np.median(v))
        mad = 1.4826 * float(np.median(np.abs(v - self.mu))) if v.size > 2 else 0.0
        self.sd = max(mad, float(v.std()), self.sd_floor, 1e-12)
        self.g = 0.0

    def z(self, x: float) -> float:
        s = (x - self.mu) / self.sd
        return -s if self.direction == "down" else s

    def update(self, x: float) -> float:
        if np.isfinite(x):
            self.g = max(0.0, self.g + self.weight * (self.z(x) - self.k))
        self.history.append(self.g)
        return self.g

    @property
    def alarm(self) -> bool:
        return self.g > self.h


def first_alarm(times: Sequence[float], values: Sequence[float], cal_mask: Sequence[bool],
                direction: str, h: float, k: float = 0.5, sd_floor: float = 0.0, weight: float = 1.0
                ) -> Tuple[Optional[float], np.ndarray]:
    """Calibrate on cal_mask windows, run CUSUM on the rest; return first alarm time."""
    times = np.asarray(times)
    values = np.asarray(values, float)
    cal_mask = np.asarray(cal_mask, bool)
    c = Cusum(direction=direction, k=k, h=h, sd_floor=sd_floor, weight=weight)
    c.calibrate(values[cal_mask])
    g = np.zeros(len(values))
    t_alarm = None
    for i in np.where(~cal_mask)[0]:
        g[i] = c.update(values[i])
        if t_alarm is None and g[i] > h:
            t_alarm = float(times[i])
    return t_alarm, g


def cusum_trace(values: np.ndarray, cal_mask: np.ndarray, direction: str, k: float = 0.5,
                sd_floor: float = 0.0, weight: float = 1.0) -> np.ndarray:
    """CUSUM statistic for every window (threshold-free), used to sweep h cheaply."""
    _, g = first_alarm(np.arange(len(values)), values, cal_mask, direction, h=np.inf, k=k, sd_floor=sd_floor, weight=weight)
    return g


def alarm_time_for_h(times: np.ndarray, g: np.ndarray, h: float) -> Optional[float]:
    idx = np.where(g > h)[0]
    return float(times[idx[0]]) if idx.size else None
