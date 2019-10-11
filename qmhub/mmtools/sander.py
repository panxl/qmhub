import os
import io
import copy
import numpy as np
import pandas as pd

from ..system import System


def load_from_file(fin, system=None, simulation=None):
    """Class to communicate with Sander.

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

    # Read fin file
    f = open(fin, 'r')
    lines = f.readlines()
    f.close()

    # Load system information
    n_qm_atoms, n_mm_atoms, n_atoms, \
        qm_charge, qm_mult, step, n_steps = \
        np.fromstring(lines[0], dtype=int, count=7, sep=' ')

    n_atoms = n_qm_atoms + n_mm_atoms

    # Load QM information
    f = io.StringIO("".join(lines[1:(n_qm_atoms + 1)]))
    qm_atoms = pd.read_csv(f, delimiter=' ', header=None, nrows=n_qm_atoms,
                           names=['pos_x', 'pos_y', 'pos_z', 'element', 'charge', 'idx']).to_records()

    if n_mm_atoms > 0:
        f = io.StringIO("".join(lines[(n_qm_atoms + 1):(n_qm_atoms + n_mm_atoms + 1)]))
        mm_atoms = pd.read_csv(f, delimiter=' ', header=None, nrows=n_mm_atoms,
                               names=['pos_x', 'pos_y', 'pos_z', 'charge', 'idx']).to_records()

    # Process QM atoms
    if np.any(qm_atoms.element.astype(str) == 'nan'):
        qm_element = np.empty(n_qm_atoms, dtype=str)
    else:
        qm_element = np.char.capitalize(qm_atoms.element.astype(str))

    # Force charge neutrality
    total_charge = qm_atoms.charge.sum() + mm_atoms.charge.sum()
    assert total_charge < 1e-2
    delta_charge = total_charge / (n_atoms - np.count_nonzero(qm_atoms.idx == -1))
    qm_atoms.charge[np.nonzero(qm_atoms.idx != -1)] -= delta_charge
    mm_atoms.charge -= delta_charge

    # Initialize System
    if system is None:
        system = System(n_atoms, n_qm_atoms, qm_charge=qm_charge, qm_mult=qm_charge)

    system.qm.atoms.positions[:] = np.vstack((qm_atoms.pos_x, qm_atoms.pos_y, qm_atoms.pos_z))
    system.qm.atoms.charges[:] = qm_atoms.charge
    system.qm.atoms.indices[:] = qm_atoms.idx
    system.qm.atoms.elements[:] = qm_element

    # Process MM atoms
    if n_mm_atoms > 0:
        system.mm.atoms.positions[:] = np.vstack((mm_atoms.pos_x, mm_atoms.pos_y, mm_atoms.pos_z))
        system.mm.atoms.charges[:] = mm_atoms.charge
        system.mm.atoms.indices[:] = mm_atoms.idx

    # Load unit cell information
    start = 1 + n_qm_atoms + n_mm_atoms
    stop = start + 3
    cell_basis = np.loadtxt(lines[start:stop], dtype=float)

    if not np.all(cell_basis == 0.0):
        system.cell_basis[:] = cell_basis

    system.qm_charge = qm_charge
    system.qm_mult = qm_mult

    if simulation is not None:
        simulation.step = step
        simulation.n_steps = n_steps

    return system
