from pathlib import Path

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from ..system import System


class IOFile(object):
    def __init__(self, cwd=None):
        self.mode = "file"
        self.cwd = cwd

    def load_system(self, input, system=None, step=None):

        self.input = Path(input)

        if self.cwd is None:
            self.cwd = self.input.parent

        if step is None:
            step = 0

        self._step = np.asarray(step)

        f = open(self.input, "r")

        # Load system information
        n_qm_atoms, n_mm_atoms, qm_charge, qm_mult, _step = np.fromfile(f, dtype="i4", count=5, sep=" ")

        # Load QM information
        dtype = [('pos_x', "f8"), ('pos_y', "f8"), ('pos_z', "f8"), ('charge', "f8"), ('element', "i4")]
        qm_atoms = np.loadtxt(f, dtype=dtype, max_rows=n_qm_atoms)

        # Load MM information
        if n_mm_atoms > 0:
            dtype = [('pos_x', "f8"), ('pos_y', "f8"), ('pos_z', "f8"), ('charge', "f8")]
            mm_atoms = np.loadtxt(f, dtype=dtype, max_rows=n_mm_atoms)

        # Load unit cell information
        cell_basis = np.loadtxt(f, max_rows=3)
        cell_basis[np.isclose(cell_basis, 0.0)] = 0.0

        f.close()

        # Initialize System
        if system is None:
            n_atoms = n_qm_atoms + n_mm_atoms
            system = System(n_atoms, n_qm_atoms, qm_charge=qm_charge, qm_mult=qm_mult)

        system.qm.atoms.positions[:] = structured_to_unstructured(qm_atoms[['pos_x', 'pos_y', 'pos_z']]).T
        system.qm.atoms.charges[:] = qm_atoms['charge']
        system.qm.atoms.elements[:] = qm_atoms['element']

        if n_mm_atoms > 0:
            system.mm.atoms.positions[:] = structured_to_unstructured(mm_atoms[['pos_x', 'pos_y', 'pos_z']]).T
            system.mm.atoms.charges[:] = mm_atoms['charge']

        if not np.all(cell_basis == 0.0):
            system.cell_basis[:] = cell_basis

        self._step[()] = _step

        return system

    def return_results(self, energy, forces, output=None):
        if output is None:
            output = self.input.with_suffix('.out')

        with open(output, 'w') as f:
            f.write("%22.14e\n" % np.asscalar(energy))
            np.savetxt(f, forces.T, fmt='%22.14e')
