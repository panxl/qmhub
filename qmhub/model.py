import numpy as np

from .electools import Elec
from .engine import Engine
from .utils import DependArray


class Model(object):
    def __init__(
        self,
        qm_positions,
        positions,
        charges,
        cell_basis,
        switching_type=None,
        cutoff=None,
        swdist=None,
        pbc=None
        ):
        """
        Creat a Model object.
        """

        self.qm_positions = qm_positions
        self.positions = positions
        self.charges = charges
        self.cell_basis = cell_basis

        self.switching_type = switching_type

        if cutoff is not None:
            self.cutoff = cutoff
        else:
            raise ValueError("cutoff is not set")

        if swdist is None:
            self.swdist = cutoff * .75
        else:
            self.swdist = swdist

        if pbc is not None:
            self.pbc = pbc
        elif np.any(self.cell_basis != 0.0):
            self.pbc = True
        else:
            self.pbc = False

        self.elec = Elec(
            self.qm_positions,
            self.positions,
            self.charges,
            self.cell_basis,
            switching_type=self.switching_type,
            cutoff=self.cutoff,
            swdist=self.swdist,
            pbc=self.pbc,
        )
