import numpy as np

from .elec import Elec
from .utils import DependArray
from .functools import get_scaling_factor, get_scaling_factor_gradient

class Model(object):
    def __init__(self, system, switching_type=None, cutoff=None, swdist=None):

        self.switching_type = switching_type

        if cutoff is not None:
            self.cutoff = cutoff
        else:
            raise ValueError("cutoff is not set")

        if swdist is None:
            self.swdist = cutoff * .75

        self.elec = Elec(
            system.qm.atoms.positions,
            system.atoms.positions,
            system.atoms.charges,
            system.cell_basis,
            switching_type=self.switching_type,
            cutoff=self.cutoff,
            swdist=self.swdist,
        )
