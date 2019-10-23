import os
import io
import copy
import numpy as np
import pandas as pd

from ..units import HARTREE_IN_KCAL_PER_MOLE, FORCE_AU_IN_IU, AMBER_AU_TO_KCAL, AMBER_FORCE_AU_TO_IU
from ..system import System


def load_from_file(fin, system=None, simulation=None):

    # Read fin file
    f = open(fin, 'r')
    lines = f.readlines()
    f.close()

    # Load system information
    n_qm_atoms, n_mm_atoms, n_link_atoms, \
        qm_charge, qm_mult, step, n_steps = \
        np.fromstring(lines[0], dtype=int, count=7, sep=' ')

    n_atoms = n_qm_atoms + n_mm_atoms

    # Load QM information
    f = io.StringIO("".join(lines[1:(n_qm_atoms + 1)]))
    qm_atoms = pd.read_csv(f, delimiter=' ', header=None, nrows=n_qm_atoms,
                           names=['pos_x', 'pos_y', 'pos_z', 'charge', 'element']).to_records()

    if n_mm_atoms > 0:
        f = io.StringIO("".join(lines[(n_qm_atoms + 1):(n_qm_atoms + n_mm_atoms + 1)]))
        mm_atoms = pd.read_csv(f, delimiter=' ', header=None, nrows=n_mm_atoms,
                               names=['pos_x', 'pos_y', 'pos_z', 'charge']).to_records()

    # Move charges from MM1 atoms to Link atoms
    qm_atoms.charge[-n_link_atoms:] = mm_atoms.charge[-n_link_atoms:]
    mm_atoms.charge[-n_link_atoms:] = 0.

    # Process QM atoms
    if np.any(qm_atoms.element.astype(str) == 'nan'):
        qm_element = np.empty(n_qm_atoms, dtype=str)
    else:
        qm_element = np.char.capitalize(qm_atoms.element.astype(str))

    # Load unit cell information
    start = 1 + n_qm_atoms + n_mm_atoms
    stop = start + 3
    cell_basis = np.loadtxt(lines[start:stop], dtype=float)

    # Initialize System
    if system is None:
        system = System(n_atoms, n_qm_atoms, qm_charge=qm_charge, qm_mult=qm_charge)

    system.qm.atoms.positions[:] = np.vstack((qm_atoms.pos_x, qm_atoms.pos_y, qm_atoms.pos_z))
    system.qm.atoms.charges[:] = qm_atoms.charge
    system.qm.atoms.elements[:] = qm_element

    if n_mm_atoms > 0:
        system.mm.atoms.positions[:] = np.vstack((mm_atoms.pos_x, mm_atoms.pos_y, mm_atoms.pos_z))
        system.mm.atoms.charges[:] = mm_atoms.charge

    if not np.all(cell_basis == 0.0):
        system.cell_basis[:] = cell_basis

    system.qm_charge = qm_charge
    system.qm_mult = qm_mult

    if simulation is not None:
        simulation.step = step
        simulation.n_steps = n_steps

    return system

def write_to_file(fout, energy, force):
    with open(fout, 'w') as f:
        f.write("%22.14e\n" % (np.asscalar(energy) / HARTREE_IN_KCAL_PER_MOLE * AMBER_AU_TO_KCAL))
        np.savetxt(f, force.T / FORCE_AU_IN_IU * AMBER_FORCE_AU_TO_IU, fmt='%22.14e')
