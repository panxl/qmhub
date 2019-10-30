import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from ..units import HARTREE_IN_KCAL_PER_MOLE, FORCE_AU_IN_IU, AMBER_AU_TO_KCAL, AMBER_FORCE_AU_TO_IU
from ..system import System


def load_from_file(fin, system=None, simulation=None, binary=True):

    f = open(fin, "rb")

    if binary:
        # Load system information
        n_qm_atoms, n_mm_atoms, n_link_atoms, \
            qm_charge, qm_mult, step = \
            np.fromfile(f, dtype="i4", count=6)

        # Load QM information
        dtype = [('pos_x', "f8"), ('pos_y', "f8"), ('pos_z', "f8"), ('charge', "f8"), ('element', "S2")]
        qm_atoms = np.fromfile(f, dtype=dtype, count=n_qm_atoms)

        if n_mm_atoms > 0:
            dtype = [('pos_x', "f8"), ('pos_y', "f8"), ('pos_z', "f8"), ('charge', "f8")]
            mm_atoms = np.fromfile(f, dtype=dtype, count=n_mm_atoms)

        # Load unit cell information
        cell_basis = np.fromfile(f, dtype="f8", count=9).reshape(3, 3)
        cell_basis[np.isclose(cell_basis, 0.0)] = 0.0
    else:
        # Load system information
        n_qm_atoms, n_mm_atoms, n_link_atoms, \
            qm_charge, qm_mult, step = \
            np.fromfile(f, dtype="i4", count=6, sep=" ")

        # Load QM information
        dtype = [('pos_x', "f8"), ('pos_y', "f8"), ('pos_z', "f8"), ('charge', "f8"), ('element', "S2")]
        qm_atoms = np.loadtxt(f, dtype=dtype, max_rows=n_qm_atoms)

        if n_mm_atoms > 0:
            dtype = [('pos_x', "f8"), ('pos_y', "f8"), ('pos_z', "f8"), ('charge', "f8")]
            mm_atoms = np.loadtxt(f, dtype=dtype, max_rows=n_mm_atoms)

        # Load unit cell information
        cell_basis = np.loadtxt(f, max_rows=3)
        cell_basis[np.isclose(cell_basis, 0.0)] = 0.0

    f.close()

    # Initialize System
    n_atoms = n_qm_atoms + n_mm_atoms

    if system is None:
        system = System(n_atoms, n_qm_atoms, qm_charge=qm_charge, qm_mult=qm_charge)

    system.qm.atoms.positions[:] = structured_to_unstructured(qm_atoms[['pos_x', 'pos_y', 'pos_z']]).T
    system.qm.atoms.charges[:] = qm_atoms['charge']
    system.qm.atoms.elements[:] = [e.decode("ascii").strip() for e in qm_atoms['element']]

    if n_mm_atoms > 0:
        system.mm.atoms.positions[:] = structured_to_unstructured(mm_atoms[['pos_x', 'pos_y', 'pos_z']]).T
        system.mm.atoms.charges[:] = mm_atoms['charge']

    if not np.all(cell_basis == 0.0):
        system.cell_basis[:] = cell_basis

    system.qm_charge = qm_charge
    system.qm_mult = qm_mult

    if simulation is not None:
        simulation.step = step

    return system


def write_to_file(fout, energy, force, binary=True):
    if binary:
        with open(fout, 'wb') as f:
            (energy / (HARTREE_IN_KCAL_PER_MOLE / AMBER_AU_TO_KCAL)).tofile(f)
            (force.T / (FORCE_AU_IN_IU / AMBER_FORCE_AU_TO_IU)).tofile(f)
    else:
        with open(fout, 'w') as f:
            f.write("%22.14e\n" % (np.asscalar(energy) / HARTREE_IN_KCAL_PER_MOLE * AMBER_AU_TO_KCAL))
            np.savetxt(f, force.T / FORCE_AU_IN_IU * AMBER_FORCE_AU_TO_IU, fmt='%22.14e')
