import os
import io
import copy
import numpy as np
import pandas as pd

from .mmbase import MMBase
from ..atomtools import QMAtoms, MMAtoms


class NAMD(MMBase):
    """Class to communicate with NAMD.

    Attributes
    ----------
    n_qm_atoms : int
        Number of QM atoms including linking atoms
    n_mm_atoms : int
        Number of MM atoms including virtual particles
    n_atoms: int
        Number of total atoms in the whole system
    qm_charge : int
        Total charge of QM subsystem
    qm_mult : int
        Multiplicity of QM subsystem
    step : int
        Current step number
    n_step : int
        Number of total steps to run in the current job

    """

    MMTOOL = 'NAMD'

    def __init__(self, fin):

        self.fin = fin

        # Read fin file
        f = open(self.fin, 'r')
        lines = f.readlines()
        f.close()

        # Load system information
        self.n_qm_atoms, self.n_mm_atoms, self.n_atoms, \
            self.qm_charge, self.qm_mult, self.step, self.n_steps = \
            np.fromstring(lines[0], dtype=int, count=7, sep=' ')

        # Load QM information
        f = io.StringIO("".join(lines[1:(self.n_qm_atoms + 1)]))
        qm_atoms = pd.read_csv(f, delimiter=' ', header=None, nrows=self.n_qm_atoms,
                               names=['pos_x', 'pos_y', 'pos_z', 'element', 'charge', 'idx']).to_records()

        if self.n_mm_atoms > 0:
            f = io.StringIO("".join(lines[(self.n_qm_atoms + 1):(self.n_qm_atoms + self.n_mm_atoms + 1)]))
            mm_atoms = pd.read_csv(f, delimiter=' ', header=None, nrows=self.n_mm_atoms,
                                   names=['pos_x', 'pos_y', 'pos_z', 'charge', 'idx', 'type', 'bonded_to_idx']).to_records()

        # Load unit cell information
        start = 1 + self.n_qm_atoms + self.n_mm_atoms
        stop = start + 3
        self.cell_basis = np.loadtxt(lines[start:stop], dtype=float)

        if np.all(self.cell_basis == 0.0):
            self.cell_basis = None

        # Process QM atoms
        self.n_virt_qm_atoms = np.count_nonzero(qm_atoms.idx == -1)
        self.n_real_qm_atoms = self.n_qm_atoms - self.n_virt_qm_atoms

        if np.any(qm_atoms.element.astype(str) == 'nan'):
            qm_element = np.empty(self.n_qm_atoms, dtype=str)
        else:
            qm_element = np.char.capitalize(qm_atoms.element.astype(str))

        # Initialize the QMAtoms object
        self.qm_atoms = QMAtoms(qm_atoms.pos_x, qm_atoms.pos_y, qm_atoms.pos_z,
                                qm_element, qm_atoms.charge, qm_atoms.idx, self.cell_basis)

        # Process MM atoms
        if self.n_mm_atoms > 0:
            self.n_virt_mm_atoms = np.count_nonzero(mm_atoms.type == 2)
            self.n_real_mm_atoms = self.n_mm_atoms - self.n_virt_mm_atoms

            real_mm_indices = np.s_[0:self.n_real_mm_atoms]
            virt_mm_indices = np.s_[self.n_real_mm_atoms:]

            orig_mm_charge = copy.copy(mm_atoms.charge[real_mm_indices])

            # Prepare for link atoms
            if self.n_virt_mm_atoms > 0:
                # Local indexes of MM1 and QM host atoms
                mm1_local_idx = np.where(mm_atoms.type == 0)[0]
                qm_host_local_idx = mm_atoms.bonded_to_idx[mm1_local_idx]

                # Local index of MM2 atoms
                mm2_local_idx = []

                for i in range(self.n_virt_qm_atoms):
                    mm2_local_idx.append(np.where((mm_atoms.type == 1) * (mm_atoms.bonded_to_idx == mm1_local_idx[i]))[0])

                # Get original MM charges
                virt_atom_mm2_idx = mm_atoms.bonded_to_idx[virt_mm_indices]
                virt_atom_mm1_idx = mm_atoms.bonded_to_idx[virt_atom_mm2_idx]
                virt_mm_charge = mm_atoms.charge[virt_mm_indices]
                np.add.at(orig_mm_charge, virt_atom_mm1_idx, virt_mm_charge)

                mm_atoms.idx[virt_mm_indices] = -1

            # Initialize the MMAtoms object
            self.mm_atoms = MMAtoms(mm_atoms.pos_x, mm_atoms.pos_y, mm_atoms.pos_z,
                                    mm_atoms.charge, mm_atoms.idx, orig_mm_charge, self.qm_atoms)

            if self.n_virt_mm_atoms > 0:
                # Get array mask to cancel 1-2 and 1-3 interactions for coulomb
                coulomb_mask = np.ones((self.n_mm_atoms, self.n_qm_atoms), dtype=bool)

                for i in range(self.n_virt_qm_atoms):
                    # Cancel 1-3 interactions between MM2 atoms and QM hosts
                    coulomb_mask[mm2_local_idx[i], qm_host_local_idx[i]] = False
                    # Cancel 1-2 interactions between MM1 atoms and QM hosts
                    coulomb_mask[mm1_local_idx[i], qm_host_local_idx[i]] = False

                # Cancel 1-3 interactions between virtual atoms and QM hosts
                # virt_atom_qm_idx = mm_atoms.bonded_to_idx[virt_atom_mm1_idx]
                for i in range(self.n_virt_qm_atoms):
                    # coulomb_mask[virt_mm_indices][virt_atom_qm_idx == qm_host_local_idx[i], qm_host_local_idx[i]] = False
                    coulomb_mask[virt_mm_indices][virt_atom_mm1_idx == mm1_local_idx[i], qm_host_local_idx[i]] = False

                self.mm_atoms.coulomb_mask = coulomb_mask

        else:
            self.mm_atoms = None

    def save_results(self):
        """Save the results of QM calculation to file."""
        fout = self.fin + ".result"
        if os.path.isfile(fout):
            os.remove(fout)

        with open(fout, 'wb') as f:
            f.write(b"%22.14e\n" % self.qm_energy)
            np.savetxt(f, np.column_stack((self.qm_force, self.qm_atoms.charge)), fmt='%22.14e')
            np.savetxt(f, self.mm_force, fmt='%22.14e')

    def preserve_input(self, n_digits=None):
        """Preserve the input file passed from NAMD."""
        import glob
        import shutil
        prev_inputs = glob.glob(self.fin + "_*")
        if prev_inputs:
            idx = max([int(i.split('_')[-1]) for i in prev_inputs]) + 1
        else:
            idx = 0

        if n_digits is not None:
            idx = '{:0{}}'.format(idx, n_digits)
        else:
            idx = str(idx)

        if os.path.isfile(self.fin):
            shutil.copyfile(self.fin, self.fin + "_" + idx)
