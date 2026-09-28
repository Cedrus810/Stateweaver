import numpy as np

from sroqm.backend import MethodSpec, PySCFBackend
from sroqm.frame import Frame


def geometry_series(mol, step, n):
    """n geometries (Bohr) moving with constant velocity `step` per frame."""
    x0 = mol.atom_coords()
    return [x0 + k * step for k in range(n)]


def converged_frames(mol, coords_list, method="rks/b3lyp"):
    be = PySCFBackend(MethodSpec(method, conv_tol=1e-10))
    frames = []
    for x in coords_list:
        m = mol.set_geom_(x, unit="Bohr", inplace=False)
        frames.append(Frame.from_outcome(m, be.run(m, with_gradient=False)))
    return frames


def lowdin_density_error(dm, frame):
    """Frobenius norm of (dm - frame.dm) in frame's orthonormal basis, summed over spins."""
    h = frame.ovlp_half
    diff = np.asarray(dm) - frame.dm
    diff = diff if diff.ndim == 3 else diff[None]
    return float(np.sqrt(sum(np.linalg.norm(h @ d @ h) ** 2 for d in diff)))


def ethylene(theta_deg, basis="6-31g"):
    """C2H4 with one CH2 group twisted by theta about the C=C axis; 0 and 180 degrees are the same molecule."""
    from pyscf import gto

    t = np.radians(theta_deg)
    cc, ch, a = 0.667, 1.085, np.radians(121.5)
    hy, hz = ch * np.sin(a), cc - ch * np.cos(a)
    atoms = [("C", (0, 0, cc)), ("C", (0, 0, -cc)), ("H", (0, hy, hz)), ("H", (0, -hy, hz))]
    for s in (1, -1):
        atoms.append(("H", (-s * hy * np.sin(t), s * hy * np.cos(t), -hz)))
    return gto.M(atom=atoms, basis=basis, verbose=0)
