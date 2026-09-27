"""Initial maximum overlap method (IMOM) across geometries.

PySCF's mom_occ keeps a fixed reference but measures overlap with the current
geometry's S. Along a trajectory the reference orbitals belong to the previous
geometry, so the overlap must use the cross matrix <chi(R_old)|chi(R_new)>.
"""
from __future__ import annotations

import numpy as np

from sroqm.backend import OccHook
from sroqm.frame import Frame, occupied_blocks
from sroqm.overlap import cross_overlap


def make_imom_hook(ref: Frame, mol) -> OccHook:
    s_cross = cross_overlap(ref.mol, mol)
    blocks = occupied_blocks(ref.mo_coeff, ref.mo_occ)
    unrestricted = ref.unrestricted

    def hook(mf):
        def get_occ(mo_energy=None, mo_coeff=None):
            if mo_coeff is None:
                mo_coeff = mf.mo_coeff
            chans = np.asarray(mo_coeff) if unrestricted else np.asarray(mo_coeff)[None]
            occ = np.zeros((chans.shape[0], chans.shape[2]))
            for s, (c_ref, occv) in enumerate(blocks):
                proj = c_ref.T @ s_cross @ chans[s]
                weight = np.einsum("ij,ij->j", proj, proj)
                occ[s, np.argsort(-weight, kind="stable")[: c_ref.shape[1]]] = occv
            return occ if unrestricted else occ[0]

        mf.get_occ = get_occ

    return hook
