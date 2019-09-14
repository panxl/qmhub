import numpy as np

from .utils import DependArray

from .atoms import Atoms
from .elec import Elec

class System(object):
    def __init__(self, n_atoms, n_qm_atoms, qm_charge, qm_mult):
        self.atoms = Atoms.new(n_atoms)

        self.qm = System.__new__(System)
        self.qm.atoms = self.atoms[:n_qm_atoms]

        n_mm_atoms = n_atoms - n_qm_atoms
        if n_mm_atoms > 0:
            self.mm = System.__new__(System)
            self.mm.atoms = self.atoms[n_qm_atoms:]
        elif n_mm_atoms == 0:
            self.mm = None
        else:
            raise ValueError("The numer of QM atoms cannot be greater than the number of total atoms.")

        self.cell_basis = DependArray(np.zeros((3, 3)), name="cell_basis")
        self.elec = Elec.new(self.qm.atoms.positions, self.atoms.positions, self.cell_basis, self.atoms._real_mask)

        self.qm.elec = self.elec[:n_qm_atoms]

        if self.mm is not None:
            self.mm.elec = self.elec[n_qm_atoms:]
 
        self.qm_charge = qm_charge
        self.qm_mult = qm_mult
