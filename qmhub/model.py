import numpy as np

from .utils import DependArray
from .functools.switching import get_near_field_mask, get_scaling_factor, get_scaling_factor_gradient

class Model(object):
    def __init__(self, system, switching_type='shift', cutoff=None, swdist=None):

        self.switching_type = switching_type

        if cutoff is not None:
            self.cutoff = cutoff
        else:
            raise ValueError("cutoff is not set")

        if swdist is None:
            self.swdist = cutoff * .75

        self.near_field_mask = DependArray(
            np.zeros(len(system.atoms), dtype=bool),
            name="near_field_mask",
            func=get_near_field_mask,
            kwargs={'cutoff': self.cutoff},
            dependencies=[system.elec.dij_min],
        )

        self.mm_charge_scaling = DependArray(
            np.zeros(len(system.atoms)),
            name="mm_charge_scaling",
            func=get_scaling_factor(switching_type),
            kwargs={'cutoff': self.cutoff, 'swdist': self.swdist},
            dependencies=[system.elec.dij_min],
        )

        self.mm_charge_scaling_gradient = DependArray(
            np.zeros((3, len(system.qm.atoms), len(system.atoms))),
            name="mm_charge_scaling_gradient",
            func=get_scaling_factor_gradient(switching_type),
            kwargs={'cutoff': self.cutoff, 'swdist': self.swdist},
            dependencies=[system.elec.dij_min, system.elec.dij_min_gradient],
        )

    # @property
    # def esp_on_qm_sites_from_mm(self):
    #     return self.system.mm.elec.dij_inverse @ self.system.mm.atoms.charges

    # @property
    # def esp_gradient_on_qm_sites_from_mm(self):
    #     return self.system.mm.elec.dij_inverse_gradient @ self.system.mm.atoms.charges
