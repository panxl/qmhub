import numpy as np

from .utils.darray import DependArray
from .atoms import Atoms


class System(object):
    '''
    QMHub System object
    '''
    def __init__(self, n_atoms, n_qm_atoms, qm_charge, qm_mult):
        '''
        Creates system in which the simulation will run. Contains the total number of atoms to run the Molecular Mechanical simulation on, 
        and the number of atoms to run the Quantum Mechanical simulation on. N.B. The numer of QM atoms cannot be greater than the number of total atoms. 
        
        Args:
            n_atoms (int):
            n_qm_atoms (int):
            qm_charge ():
            qm_mult ():
        '''
        self.qm_index = np.s_[:n_qm_atoms]
        self.mm_index = np.s_[n_qm_atoms:]

        self.atoms = Atoms.new(n_atoms)

        self.qm = System.__new__(System)
        self.qm.atoms = self.atoms[self.qm_index]

        if n_atoms > n_qm_atoms:
            self.mm = System.__new__(System)
            self.mm.atoms = self.atoms[self.mm_index]
        elif n_atoms == n_qm_atoms:
            self.mm = None
        else:
            raise ValueError("The numer of QM atoms cannot be greater than the number of total atoms.")

        self.cell_basis = DependArray(np.zeros((3, 3)), name="cell_basis")
 
        self.qm_charge = qm_charge
        self.qm_mult = qm_mult

    def wrap_positions(self):
        '''what
        
        Args:
            
        '''
        self.atoms.positions -= self.qm.atoms.positions.mean(axis=1, keepdims=True)
        if not np.all(self.cell_basis == 0):
            self.atoms.positions -= np.around(self.atoms.positions / np.diagonal(self.cell_basis)[:, np.newaxis]) * np.diagonal(self.cell_basis)[:, np.newaxis]
