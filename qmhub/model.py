import numpy as np

from .utils import DependArray
from .func import get_scaling_factor, get_scaling_factor_gradient

class Model(object):
    def __init__(self, system, switching_type='shift', cutoff=None, swdist=None, pbc=None):

        self.system = system
        self.switching_type = switching_type
        self.pbc = pbc

        if self.pbc is None:
            if self.system.cell_basis is not None:
                self.pbc = True

        # self.near_field_mask = (system.mm.elecs.dij_min < cutoff)
        # self.mm_near_charges = system.mm.atoms.charges[self.near_field_mask]

        if cutoff is not None:
            self.cutoff = DependArray([cutoff], name="cutoff")

        if swdist is not None:
            self.swdist = DependArray([cutoff], name="swdist")

        self.mm_charge_scaling = DependArray(
            np.zeros(len(system.atoms)),
            name="mm_charge_scaling",
            func=get_scaling_factor,
            dependencies=[self.cutoff, system.elecs.dij_min],
        )

    # @property
    # def esp_on_qm_sites_from_mm(self):
    #     return self.system.mm.elecs.dij_inverse @ self.system.mm.atoms.charges

    # @property
    # def esp_gradient_on_qm_sites_from_mm(self):
    #     return self.system.mm.elecs.dij_inverse_gradient @ self.system.mm.atoms.charges