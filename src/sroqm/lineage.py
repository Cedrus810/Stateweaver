"""An electronic-state lineage: one state's continuous history along the nuclear path (spec §6.2)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class LineageStatus(Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    PRUNED = "pruned"


@dataclass
class Lineage:
    id: int
    multiplicity: int
    max_frames: int = 6
    parent_id: int | None = None
    birth_step: int = 0
    status: LineageStatus = LineageStatus.ACTIVE
    fingerprints: list = field(default_factory=list)
    energies: list = field(default_factory=list)
    gaps: list = field(default_factory=list)  # per-spin frontier gaps per step
    frames: deque = field(init=False)

    def __post_init__(self):
        if self.max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        self.frames = deque(maxlen=self.max_frames)

    def push(self, frame, fingerprint, gaps: tuple[float, ...] = ()) -> None:
        self.frames.appendleft(frame)
        self.fingerprints.append(fingerprint)
        self.energies.append(frame.energy)
        self.gaps.append(tuple(gaps))

    def cut_history(self) -> None:
        """Drop all frames but the latest, e.g. after a root switch; per-step records are kept."""
        latest = self.frames[0]
        self.frames.clear()
        self.frames.append(latest)

    @property
    def history(self) -> tuple:
        return tuple(self.frames)

    @property
    def last_frame(self):
        return self.frames[0] if self.frames else None

    @property
    def last_fingerprint(self):
        return self.fingerprints[-1] if self.fingerprints else None

    @property
    def last_gaps(self) -> tuple[float, ...] | None:
        return self.gaps[-1] if self.gaps else None
