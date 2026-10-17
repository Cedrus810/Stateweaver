import dataclasses

import pytest

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.frame import Frame
from sroqm.tolerances import Tolerances


def test_defaults_are_the_spec_values():
    t = Tolerances()
    assert (t.conv_tol, t.same_state_sv, t.admit_de, t.deg_band, t.accept_de) == (1e-7, 0.9, 1e-5, 1e-4, 5e-5)


def test_tolerances_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Tolerances().conv_tol = 1e-6


@pytest.mark.parametrize("kw", [dict(conv_tol=0), dict(admit_de=-1e-5), dict(deg_band=0), dict(accept_de=0),
                                dict(same_state_sv=0), dict(same_state_sv=1.5)])
def test_tolerances_validate(kw):
    with pytest.raises(ValueError):
        Tolerances(**kw)


def test_same_state(water, excited_water):
    out = PySCFBackend(MethodSpec("uks/b3lyp")).run(water, with_gradient=False)
    ground = Frame.from_outcome(water, out)
    t = Tolerances()
    assert t.same_state(ground, ground)
    assert not t.same_state(ground, excited_water)
