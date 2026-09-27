import pytest

from sroqm.lineage import Lineage, LineageStatus


class F:  # minimal stand-in for Frame: Lineage only reads .energy
    def __init__(self, e):
        self.energy = e


def test_push_orders_history_most_recent_first_and_caps_frames():
    lin = Lineage(id=0, multiplicity=1, max_frames=2)
    assert lin.last_frame is None and lin.last_fingerprint is None and lin.history == ()
    for e in (1.0, 2.0, 3.0):
        lin.push(F(e), f"fp{e}")
    assert [f.energy for f in lin.history] == [3.0, 2.0]
    assert lin.energies == [1.0, 2.0, 3.0]
    assert lin.last_fingerprint == "fp3.0"
    assert lin.status is LineageStatus.ACTIVE


def test_max_frames_must_be_positive():
    with pytest.raises(ValueError):
        Lineage(id=0, multiplicity=1, max_frames=0)
