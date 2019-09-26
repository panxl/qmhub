import numpy as np

from .electools import Elec
from .engine import Engine
from .utils import DependArray


class Model(object):
    def __init__(self, system, switching_type=None, cutoff=None, swdist=None, pbc=None):

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
        elif np.any(system.cell_basis != 0.0):
            self.pbc = True
        else:
            self.pbc = False

        self.elec = Elec(
            system.qm.atoms.positions,
            system.atoms.positions,
            system.atoms.charges,
            system.cell_basis,
            switching_type=self.switching_type,
            cutoff=self.cutoff,
            swdist=self.swdist,
            pbc=self.pbc,
        )

        self.engine = Engine(
            system.qm.atoms.positions,
            system.qm.atoms.elements,
            self.elec.embedding_mm_positions,
            self.elec.embedding_mm_charges,
            charge=system.qm_charge,
            mult=system.qm_mult,
        )
