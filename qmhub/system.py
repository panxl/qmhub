import numpy as np

from .utils import DependArray

from .atoms import Atoms
from .elec import Elecs

class System(object):
    def __init__(self, n_atoms, n_qm_atoms):
        self.atoms = Atoms(n_atoms)

        self.qm = System.__new__(System)
        self.qm.atoms = self.atoms[:n_qm_atoms]
        self.qm.total_charge = 0
        self.qm.mult = 1

        n_mm_atoms = n_atoms - n_qm_atoms
        if n_mm_atoms > 0:
            self.mm = System.__new__(System)
            self.mm.atoms = self.atoms[n_qm_atoms:]
        elif n_mm_atoms == 0:
            self.mm = None
        else:
            raise ValueError("The numer of QM atoms cannot be greater than the number of total atoms.")

        self.cell_basis = DependArray(np.zeros((3, 3)), name="cell_basis")
        self.elecs = Elecs(self.qm.atoms.positions, self.atoms.positions, self.cell_basis)
 
        # self.qm.elecs = self.elecs[:n_qm_atoms]
        # self.qm.elecs = copy(self.elecs)
        # self.qm.elecs.real = copy(self.qm.elecs)
        # self.qm.elecs.virtual = copy(self.qm.elecs.real)

        # if self.mm is not None:
        #     self.mm.elecs = copy(self.elecs)
        #     self.mm.elecs.real = copy(self.mm.elecs)
        #     self.mm.elecs.virtual = copy(self.mm.elecs.real)