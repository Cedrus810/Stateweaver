"""Per-lineage coordination monitor (Stateweaver design docs/coordination-electronic-lineage-research-design.zh-CN.md
§6, §7; WP2): stable labels, transition labels and evidence quality kept apart, with hysteresis and persistent
confirmation, a separate history per (lineage, pair, segment), and events that can be written as JSON lines and
replayed from a saved state.

Bypass observer: it reads bonding.BondObservation's (and the beam's lifecycle events); it never changes energies,
forces, state selection or search.

Labels (design §6.1). Stable: BONDED / UNBOUND (None until initialised). Transition: BREAKING / FORMING / AMBIGUOUS.
Events: INITIALIZED, DISSOCIATED (BONDED -> UNBOUND), FORMED (UNBOUND -> BONDED), ATTEMPT_WITHDRAWN (a departure that
came back), SEGMENT_RESTART (revived or uncertain lineage: history kept, label must be re-confirmed), MONITOR_CLOSED
(lineage merged / evicted), SELECTED_LINEAGE_SWITCH (descriptor differences between the two lineages, histories not
spliced), STATE_DEPENDENT_BONDING_CHANGE (lineages at the same geometry disagree on a pair's stable label).

Hysteresis (design §6.2), per donor class, every value from calibration (none built in):
    C_on  = (r <= r_on)  and (M >= m_on)
    C_off = (r >= r_off) and (M <= m_off),     r_on < r_off, m_on > m_off
A stable label is (re)committed after n_confirm consecutive valid observations in its region; an invalid observation
(unconverged, missing) interrupts the count and leaves the last label marked stale. The confirmation count is in
observations, not path length or time: its step-size dependence must be checked per experiment (design §6.2).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

from sroqm.bonding import BondObservation, observe_beam

UNCERTAIN_ALARMS = ("overlap_drop", "imom_fallback")  # lineage continuity in doubt -> new monitor segment (§6.3)
CLOSING_EVENTS = ("merge", "evict")                     # the lineage leaves the beam for good


@dataclass(frozen=True)
class BondThresholds:
    r_on: float      # Angstrom
    r_off: float
    m_on: float      # Mayer bond order
    m_off: float
    n_confirm: int = 2

    def __post_init__(self):
        if not (self.r_on < self.r_off and self.m_on > self.m_off):
            raise ValueError("need r_on < r_off and m_on > m_off")
        if self.n_confirm < 1:
            raise ValueError("n_confirm must be >= 1")


def region(obs: BondObservation, th: BondThresholds) -> str:
    """on / off / mid / conflict (distance and bond order point to opposite sides)."""
    r, m = obs.distance_angstrom, obs.mayer_order
    if r <= th.r_on and m >= th.m_on:
        return "on"
    if r >= th.r_off and m <= th.m_off:
        return "off"
    if (r <= th.r_on and m <= th.m_off) or (r >= th.r_off and m >= th.m_on):
        return "conflict"
    return "mid"


@dataclass
class PairMonitor:
    """Label state of one pair on one lineage segment."""
    lineage_id: int
    pair_id: str
    segment: int = 0
    stable: str | None = None
    transition: str | None = "AMBIGUOUS"
    quality: str = "VALID"
    target: str | None = None     # the stable label being confirmed
    count: int = 0                # consecutive valid observations supporting target
    onset: int | None = None      # step the current departure / initialisation candidate began
    last_valid: int | None = None
    last_r: float | None = None
    last_m: float | None = None

    def _event(self, kind, step, **kw):
        return dict(event_type=kind, step=step, lineage_id=self.lineage_id, segment=self.segment,
                    pair_id=self.pair_id, **kw)

    def update(self, obs: BondObservation | None, valid: bool, step: int, th: BondThresholds) -> list[dict]:
        if obs is None or not valid:
            self.count, self.quality = 0, "UNCONVERGED" if obs is not None else "MISSING"
            return []
        events, reg = [], region(obs, th)
        r, m = obs.distance_angstrom, obs.mayer_order
        self.quality = "DESCRIPTOR_CONFLICT" if reg == "conflict" else "VALID"
        if self.stable is None:  # initialisation: never assumed BONDED (§6.2 rule 1)
            tgt = {"on": "BONDED", "off": "UNBOUND"}.get(reg)
            if tgt is None:
                self.target, self.count, self.onset = None, 0, None
            else:
                if tgt != self.target:
                    self.target, self.count, self.onset = tgt, 0, step
                self.count += 1
                if self.count >= th.n_confirm:
                    events.append(self._event("INITIALIZED", step, previous_label=None, new_label=tgt,
                                              onset_step=self.onset, confirmed_at=step, source_type="GEOMETRY"))
                    self.stable, self.transition, self.target, self.count, self.onset = tgt, None, None, 0, None
        else:
            bonded = self.stable == "BONDED"
            home, away = ("on", "off") if bonded else ("off", "on")
            go, move = ("UNBOUND", "BREAKING") if bonded else ("BONDED", "FORMING")
            trend = self.last_r is not None and ((r > self.last_r and m < self.last_m) if bonded
                                                 else (r < self.last_r and m > self.last_m))
            if reg == home:
                if self.onset is not None:
                    events.append(self._event("ATTEMPT_WITHDRAWN", step, label=self.stable, onset_step=self.onset,
                                              source_type="GEOMETRY"))
                self.transition, self.target, self.count, self.onset = None, None, 0, None
            elif reg == away:
                if self.onset is None:
                    self.onset = step
                self.transition, self.target = move, go
                self.count += 1  # home / mid / conflict reset it, so these are consecutive
                if self.count >= th.n_confirm:
                    events.append(self._event("DISSOCIATED" if bonded else "FORMED", step, previous_label=self.stable,
                                              new_label=go, onset_step=self.onset, confirmed_at=step,
                                              source_type="GEOMETRY"))
                    self.stable, self.transition, self.target, self.count, self.onset = go, None, None, 0, None
            else:  # mid or conflict: keep the stable label; no endless BREAKING without a trend (§6.2 rule 4)
                if self.onset is None:
                    self.onset = step
                self.count = 0
                self.transition = move if reg == "mid" and trend else "AMBIGUOUS"
        self.last_valid, self.last_r, self.last_m = step, r, m
        return events


class CoordinationMonitor:
    """All (lineage, pair) monitors along one path. `pairs`: bonding.PairSpec list; `thresholds`: {donor_class:
    BondThresholds} (a pair's donor_class must be present)."""

    def __init__(self, pairs, thresholds: dict[str, BondThresholds]):
        missing = {p.donor_class for p in pairs} - set(thresholds)
        if missing:
            raise ValueError(f"no thresholds for donor class(es) {sorted(missing)}")
        self.pairs, self.thresholds = list(pairs), dict(thresholds)
        self.monitors: dict[tuple[int, str], PairMonitor] = {}
        self.closed: list[PairMonitor] = []
        self.selected: int | None = None
        self.disagreeing: set[str] = set()
        self.last_obs: dict[int, dict[str, BondObservation]] = {}

    def _th(self, pair_id):
        return self.thresholds[next(p.donor_class for p in self.pairs if p.pair_id == pair_id)]

    def _restart(self, lid, step, reason):
        events = []
        for p in self.pairs:
            old = self.monitors.get((lid, p.pair_id))
            seg = 0 if old is None else old.segment + 1
            if old is not None:
                self.closed.append(old)
                events.append(dict(event_type="SEGMENT_RESTART", step=step, lineage_id=lid, segment=seg,
                                   pair_id=p.pair_id, reason=reason, previous_label=old.stable))
            self.monitors[(lid, p.pair_id)] = PairMonitor(lid, p.pair_id, seg)
        return events

    def step(self, step: int, observations: dict[int, list[BondObservation]], selected: int, events=(),
             invalid=(), uncertain=()) -> list[dict]:
        """observations: {lineage id: observations of this step} for the ACTIVE lineages; events: the beam's
        (kind, lineage id) events of this step; invalid: lineages whose frame is not a converged SCF; uncertain:
        lineages whose continuity is in doubt (UNCERTAIN_ALARMS)."""
        out = []
        for kind, lid in events:
            if lid is None:
                continue
            if kind == "revive" and any(k[0] == lid for k in self.monitors):
                out += self._restart(lid, step, "revived")
            elif kind in CLOSING_EVENTS:
                for p in self.pairs:
                    mon = self.monitors.pop((lid, p.pair_id), None)
                    if mon is not None:
                        self.closed.append(mon)
                        out.append(dict(event_type="MONITOR_CLOSED", step=step, lineage_id=lid, segment=mon.segment,
                                        pair_id=p.pair_id, reason=kind, last_label=mon.stable))
        for lid in uncertain:
            if any(k[0] == lid for k in self.monitors):
                out += self._restart(lid, step, "lineage_uncertain")
        for lid, obs in observations.items():
            by_pair = {o.pair_id: o for o in obs}
            for p in self.pairs:
                mon = self.monitors.setdefault((lid, p.pair_id), PairMonitor(lid, p.pair_id))
                out += mon.update(by_pair.get(p.pair_id), lid not in invalid, step, self._th(p.pair_id))
        for (lid, pid), mon in self.monitors.items():
            if lid not in observations:  # lost / dormant this step: label kept, marked stale
                mon.update(None, False, step, self._th(pid))
        if self.selected is not None and selected != self.selected:
            old, new = self.last_obs.get(self.selected, {}), {o.pair_id: o for o in observations.get(selected, [])}
            diff = {}
            for p in self.pairs:
                a, b = old.get(p.pair_id), new.get(p.pair_id)
                if a is not None and b is not None:
                    diff[p.pair_id] = dict(dmayer=b.mayer_order - a.mayer_order,
                                           dcharge=[b.charges[i] - a.charges[i] for i in range(2)],
                                           dspin=[b.spins[i] - a.spins[i] for i in range(2)],
                                           labels=[self._label(self.selected, p.pair_id), self._label(selected, p.pair_id)])
            out.append(dict(event_type="SELECTED_LINEAGE_SWITCH", step=step, from_lineage=self.selected,
                            to_lineage=selected, source_type="ELECTRONIC", differences=diff,
                            note="previous lineage's last observation" if self.selected not in observations else ""))
        for p in self.pairs:  # same geometry, different branches, different stable labels
            labels = {lid: self._label(lid, p.pair_id) for lid in observations}
            known = {lid: lab for lid, lab in labels.items() if lab is not None}
            if len(set(known.values())) > 1:
                if p.pair_id not in self.disagreeing:
                    self.disagreeing.add(p.pair_id)
                    out.append(dict(event_type="STATE_DEPENDENT_BONDING_CHANGE", step=step, pair_id=p.pair_id,
                                    labels=known, source_type="ELECTRONIC"))
            else:
                self.disagreeing.discard(p.pair_id)
        self.selected = selected
        for lid, obs in observations.items():
            self.last_obs[lid] = {o.pair_id: o for o in obs}
        return out

    def _label(self, lid, pid):
        mon = self.monitors.get((lid, pid))
        return None if mon is None else mon.stable

    def snapshot(self) -> dict:
        return {f"{lid}:{pid}": dict(stable=m.stable, transition=m.transition, quality=m.quality, segment=m.segment,
                                     last_valid=m.last_valid, pending_since=m.onset, confirmation_progress=m.count)
                for (lid, pid), m in sorted(self.monitors.items())}

    def to_state(self) -> dict:
        """Everything step() depends on, JSON-able: restoring it continues with the same events (design §7)."""
        return dict(monitors=[dataclasses.asdict(m) for m in self.monitors.values()],
                    closed=[dataclasses.asdict(m) for m in self.closed], selected=self.selected,
                    disagreeing=sorted(self.disagreeing),
                    last_obs={str(lid): {pid: dataclasses.asdict(o) for pid, o in obs.items()}
                              for lid, obs in self.last_obs.items()})

    @classmethod
    def from_state(cls, pairs, thresholds, state: dict) -> "CoordinationMonitor":
        mon = cls(pairs, thresholds)
        for d in state["monitors"]:
            m = PairMonitor(**d)
            mon.monitors[(m.lineage_id, m.pair_id)] = m
        mon.closed = [PairMonitor(**d) for d in state["closed"]]
        mon.selected, mon.disagreeing = state["selected"], set(state["disagreeing"])
        mon.last_obs = {int(lid): {pid: BondObservation(**{**o, "charges": tuple(o["charges"]),
                                                             "spins": tuple(o["spins"]), "quality": tuple(o["quality"])})
                                   for pid, o in obs.items()} for lid, obs in state["last_obs"].items()}
        return mon


def beam_step(monitor: CoordinationMonitor, result, step: int) -> list[dict]:
    """Feed one BeamResult: observe every ACTIVE lineage, mark unconverged and continuity-doubtful ones (§6.3)."""
    obs = observe_beam(result, monitor.pairs, step)
    invalid = {lid for lid, r in result.results.items() if lid in obs and not r.converged}
    uncertain = {lid for lid, r in result.results.items()
                 if lid in obs and any(a.name in UNCERTAIN_ALARMS for a in r.alarms)}
    return monitor.step(step, obs, result.lineage_id, result.events, invalid, uncertain)


def write_jsonl(path, events) -> None:
    with open(path, "a") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
