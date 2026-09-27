import math
from typing import Optional, Sequence
import numpy as np

class TrackGeometry:
    """Преобразование продольной координаты пути s в координаты x, y, yaw."""
    def __init__(self, points: Optional[Sequence[Sequence[float]]] = None):
        if points is None or len(points) < 2:
            self.points = None
            self.cum = None
            return

        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("Track points must be Nx2 array-like.")

        diffs = np.diff(self.points, axis=0)
        seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
        self.cum = np.concatenate(([0.0], np.cumsum(seg_len)))

    def _heading(self, idx: int) -> float:
        if self.points is None:
            return 0.0
        idx = max(0, min(idx, len(self.points) - 2))
        p0 = self.points[idx]
        p1 = self.points[idx + 1]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        if abs(dx) < 1.0e-9 and abs(dy) < 1.0e-9:
            return 0.0
        return math.atan2(dy, dx)

    def pose_at(self, s: float):
        s = float(s)
        if self.points is None:
            return s, 0.0, 0.0

        total = float(self.cum[-1])
        if s <= 0.0:
            p = self.points[0]
            return float(p[0]), float(p[1]), self._heading(0)
        if s >= total:
            p = self.points[-1]
            return float(p[0]), float(p[1]), self._heading(len(self.points) - 2)

        idx = int(np.searchsorted(self.cum, s, side="right") - 1)
        idx = max(0, min(idx, len(self.points) - 2))
        seg_len = float(self.cum[idx + 1] - self.cum[idx])

        if seg_len <= 1.0e-9:
            p = self.points[idx]
            return float(p[0]), float(p[1]), self._heading(idx)

        ratio = (s - float(self.cum[idx])) / seg_len
        p0 = self.points[idx]
        p1 = self.points[idx + 1]
        x = float(p0[0] + ratio * (p1[0] - p0[0]))
        y = float(p0[1] + ratio * (p1[1] - p0[1]))
        yaw = self._heading(idx)
        return x, y, yaw
