import weakref
import numpy as np

import qmhub.helpmelib as pme
from .dobject import cache_update


class DependPME(pme.PMEInstanceD):
    def __init__(self, cell_basis, alpha, order, nfft):
        super().__init__()

        self._name = "PME"
        self._kwargs = {"alpha": alpha, "order": order, "nfft": nfft}
        self._dependencies = [cell_basis]
        self._dependants = []
        self._cache_valid = False

    def _func(self, cell_basis, alpha, order, nfft):
        super().setup(
                1,
                alpha.item(),
                order,
                *nfft.tolist(),
                1.,
                1,
        )

        super().set_lattice_vectors(
            *np.diag(cell_basis).tolist(),
            *[90., 90., 90.],
            self.LatticeType.XAligned,
        )

    @cache_update
    def compute_recip_esp(self, positions, grid_positions, grid_charges):
        recip_esp = np.zeros((len(positions.T), 4))

        charges = np.ascontiguousarray(grid_charges)[:, np.newaxis]
        coord1 = np.ascontiguousarray(grid_positions.T)
        coord2 = np.ascontiguousarray(positions.T)
        mat = pme.MatrixD
        super().compute_P_rec(
            0, 
            mat(charges),
            mat(coord1),
            mat(coord2),
            1,
            mat(recip_esp),
        )

        return np.ascontiguousarray(recip_esp.T)

    def add_dependant(self, dependant):
        self._dependants.append(weakref.ref(dependant))

    def update_cache(self):
        if not self._cache_valid:
            self._func(*self._dependencies, **self._kwargs)
            self._cache_valid = True
