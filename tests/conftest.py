import numpy as np
import pytest
from pyscf import gto, lib

lib.num_threads(4)

WATER = "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"
O2 = "O 0 0 0; O 0 0 1.21"


@pytest.fixture(scope="session")
def water():
    return gto.M(atom=WATER, basis="6-31g", verbose=0)


@pytest.fixture(scope="session")
def oxygen():
    return gto.M(atom=O2, basis="6-31g", spin=2, verbose=0)


@pytest.fixture(scope="session")
def water_step():
    """Per-step displacement in Bohr: symmetric O-H stretch, about one 0.5 fs MD step."""
    d = np.zeros((3, 3))
    d[1] = [0.0, 0.02, 0.012]
    d[2] = [0.0, -0.02, 0.012]
    return d


@pytest.fixture(scope="session")
def excited_water(water):
    """UKS water with the alpha HOMO->LUMO excitation, converged with PySCF's MOM at the template geometry."""
    from pyscf import dft
    from pyscf.scf import addons

    from sroqm.frame import Frame

    ground = dft.UKS(water, xc="b3lyp")
    ground.kernel()
    occ = ground.mo_occ.copy()
    homo = int(occ[0].sum()) - 1
    occ[0][homo], occ[0][homo + 1] = 0.0, 1.0
    mf = addons.mom_occ(dft.UKS(water, xc="b3lyp"), ground.mo_coeff, occ)
    mf.kernel(dm0=mf.make_rdm1(ground.mo_coeff, occ))
    assert mf.converged
    return Frame.from_orbitals(water, mf.mo_coeff, mf.mo_occ, mf.e_tot)
