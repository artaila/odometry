from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class StampedSample:
    stamp: float
    msg: Any

@dataclass
class SyncMatch:
    stamp: float
    samples: Dict[str, StampedSample]

class SyncBuffer:
    """Буфер приблизительной временной синхронизации (окно ±50 мс)."""
    def __init__(
        self,
        topics: List[str],
        window_sec: float = 0.05,
        max_age_sec: float = 0.5,
        min_interval_sec: float = 0.0,
    ):
        self.topics = topics
        self.window = float(window_sec)
        self.max_age = float(max_age_sec)
        self.min_interval = float(min_interval_sec)

        self.buffers: Dict[str, deque] = {topic: deque() for topic in topics}
        self.last_emitted_stamp = -1.0e18

    def clear(self):
        for dq in self.buffers.values():
            dq.clear()
        self.last_emitted_stamp = -1.0e18

    def add(self, topic: str, stamp: float, msg: Any) -> None:
        if topic not in self.buffers:
            return
        self.buffers[topic].append(StampedSample(stamp=float(stamp), msg=msg))
        self._purge(float(stamp))

    def _purge(self, now: float) -> None:
        for dq in self.buffers.values():
            while dq and now - dq[0].stamp > self.max_age:
                dq.popleft()

    def _nearest(self, topic: str, ref: float) -> Optional[StampedSample]:
        best: Optional[StampedSample] = None
        best_err = self.window + 1.0

        for sample in self.buffers[topic]:
            err = abs(sample.stamp - ref)
            if err <= self.window and err < best_err:
                best = sample
                best_err = err
                if err < 1.0e-9:
                    break
        return best

    def latest_frame(self, trigger_topic: str) -> Optional[SyncMatch]:
        if trigger_topic not in self.buffers:
            return None

        dq = self.buffers[trigger_topic]
        if not dq:
            return None

        for sample in reversed(dq):
            if sample.stamp <= self.last_emitted_stamp:
                break
            if sample.stamp - self.last_emitted_stamp < self.min_interval:
                return None

            matches = {trigger_topic: sample}
            ok = True

            for topic in self.topics:
                if topic == trigger_topic:
                    continue
                match = self._nearest(topic, sample.stamp)
                if match is None:
                    ok = False
                    break
                matches[topic] = match

            if ok:
                self.last_emitted_stamp = sample.stamp
                return SyncMatch(stamp=sample.stamp, samples=matches)

        return None
