import math
import numpy as np

from .functools import (
    get_rij,
    get_dij,
    get_dij_gradient,
    get_dij_inverse,
    get_dij_inverse_gradient,
    get_dij_min,
    get_dij_min_gradient,
)
from .functools import Ewald
from .utils import DependArray


class Elec(object):

    _darray = ['rij', 'dij', 'dij_gradient', 'dij_inverse', 'dij_inverse_gradient',
               'dij_min', 'dij_min_gradient', 'ewald']

    def __init__(self, _real_mask=None, **kwargs):
        for key in Elec._darray:
            value = kwargs[key]
            setattr(self, key, value)

        self._real_mask = _real_mask

    @classmethod
    def new(cls, ri, rj, charges, cell_basis, _real_mask):
        rij = DependArray(
            name="rij",
            func=get_rij,
            dependencies=[ri, rj],
        )
        dij = DependArray(
            name="dij",
            func=get_dij,
            dependencies=[rij],
        )
        dij_gradient = DependArray(
            name="dij_gradient",
            func=get_dij_gradient,
            dependencies=[rij, dij],
        )
        dij_inverse = DependArray(
            name="dij_inverse",
            func=get_dij_inverse,
            dependencies=[dij],
        )
        dij_inverse_gradient = DependArray(
            name="dij_inverse_gradient",
            func=get_dij_inverse_gradient,
            dependencies=[dij_inverse, dij_gradient],
        )
        dij_min = DependArray(
            name="dij_min",
            func=get_dij_min,
            dependencies=[dij_inverse],
        )
        dij_min_gradient = DependArray(
            name="dij_min_gradient",
            func=get_dij_min_gradient,
            dependencies=[dij_min, dij_inverse, dij_inverse_gradient],
        )

        ewald = Ewald(rij, charges, cell_basis)

        kwargs = {}
        for key in Elec._darray:
            kwargs[key] = locals()[key]
        kwargs['_real_mask'] = _real_mask
        kwargs['ewald'] = ewald
        return cls(**kwargs)

    @classmethod
    def from_elec(cls, elec, index=None):
        kwargs = {key: getattr(elec, key)[..., index] for key in Elec._darray}
        kwargs['_real_mask'] = elec._real_mask[..., index]
        return cls(**kwargs)

    def __len__(self):
        return self.rij.shape[2]

    def __iter__(self):
        for index in range(len(self)):
            yield self.from_elec(self, index)

    def __getitem__(self, index):
        return self.from_elec(self, index)

    @property
    def real(self):
        return self[self._real_mask.view(np.ndarray)]

    @property
    def virtual(self):
        return self[~self._real_mask.view(np.ndarray)]
