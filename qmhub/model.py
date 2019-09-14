import numpy as np

from .utils import DependArray
from .func import get_scaling_factor, get_scaling_factor_gradient

class Model(object):
    def __init__(self, system, switching_type='shift', cutoff=None, swdist=None, pbc=None):

        self.switching_type = switching_type
        self.pbc = pbc

        if self.pbc is None:
            if system.cell_basis is not None:
                self.pbc = True

        if cutoff is not None:
            self.cutoff = DependArray([cutoff], name="cutoff")

        if swdist is not None:
            self.swdist = DependArray([swdist], name="swdist")

        self.near_field_mask = DependArray(
            np.zeros(len(system.atoms), dtype=bool),
            name="near_field_mask",
            func=(lambda x, y: x < y),
            dependencies=[system.elec.dij_min, self.cutoff],
        )

        self.mm_charge_scaling = DependArray(
            np.zeros(len(system.atoms)),
            name="mm_charge_scaling",
            func=get_scaling_factor,
            dependencies=[self.cutoff, self.swdist, system.elec.dij_min],
        )

    # @property
    # def esp_on_qm_sites_from_mm(self):
    #     return self.system.mm.elec.dij_inverse @ self.system.mm.atoms.charges

    # @property
    # def esp_gradient_on_qm_sites_from_mm(self):
    #     return self.system.mm.elec.dij_inverse_gradient @ self.system.mm.atoms.charges
