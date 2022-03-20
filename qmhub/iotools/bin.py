from pathlib import Path

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from ..system import System


class IOBin(object):
    def __init__(self, cwd=None):
        self.mode = "bin"
        self.cwd = cwd

    def load_system(self, input, system=None, step=None):

        self.input = Path(input)
        self.cwd = self.cwd or self.input.parent

        if step is None:
            step = 0

        self._step = np.asarray(step)

        f = open(self.input, "rb")

        # Load system information
        n_qm_atoms, n_mm_atoms, qm_charge, qm_mult, _step = np.fromfile(f, dtype="i4", count=5)

        # Load QM information
        dtype = [('pos_x', "f8"), ('pos_y', "f8"), ('pos_z', "f8"), ('charge', "f8"), ('element', "i4")]
        qm_atoms = np.fromfile(f, dtype=dtype, count=n_qm_atoms)

        # Load MM information
        if n_mm_atoms > 0:
            dtype = [('pos_x', "f8"), ('pos_y', "f8"), ('pos_z', "f8"), ('charge', "f8")]
            mm_atoms = np.fromfile(f, dtype=dtype, count=n_mm_atoms)

        # Load unit cell information
        cell_basis = np.fromfile(f, dtype="f8", count=9).reshape(3, 3)
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

        system.wrap_positions()

        self._step[()] = _step

        return system

    def return_results(self, energy, forces, output=None):
        output = output or self.input.with_suffix('.out')

        with open(output, 'wb') as f:
            energy.tofile(f)
            forces.T.tofile(f)

    @staticmethod
    def save_input(input):
        """Preserve the input file passed from the driver."""
        import glob
        import shutil

        input = str(input)
        prev_inputs = glob.glob(input + "_*")

        if prev_inputs:
            idx = max([int(i.split('_')[-1]) for i in prev_inputs]) + 1
        else:
            idx = 0

        if Path(input).is_file():
            shutil.copyfile(input, f"{input}_{idx:04d}")
