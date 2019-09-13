import numpy as np

from .utils import DependArray

class Atom(object):

    def __init__(self, atoms, index):

        self.positions = atoms.positions[:, index]
        self.charges = atoms.charges[index]
        self.indices = atoms.indices[index]
        self.elements = atoms.elements[index]

    def __len__(self):
        return len(self.indices)

class Atoms(object):
    def __init__(self, n_atoms, atoms=None):
        self.n_atoms = n_atoms

        if atoms is None:
            self.positions = DependArray(np.zeros((3, n_atoms)))
            self.charges = DependArray(np.zeros(n_atoms))
            self.indices = DependArray(np.zeros(n_atoms, dtype=int))
            self.elements = DependArray(np.zeros(n_atoms, dtype=str))
        else:
            self.positions = atoms.positions
            self.charges = atoms.charges
            self.indices = atoms.indices
            self.elements = atoms.elements

        self._uptodate = np.zeros(1, dtype=bool)

    def __len__(self):
        return self.n_atoms

    def __iter__(self):
        for index in range(len(self)):
            yield Atom(self, index)

    def __getitem__(self, index):
        return Atom(self, index)

    def __setitem__(self, index, value):
        atom = Atom(self, index)
        atom.positions = value.positions
        atom.charges = value.charges
        atom.indices = value.indices
        atom.elements = value.elements

    @property
    def _n_real_atoms(self):
        return np.count_nonzero(self.indices != -1)

    @property
    def real(self):
        return self[:self._n_real_atoms]

    @property
    def virtual(self):
        return self[self._n_real_atoms:]