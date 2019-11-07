import os, stat, time
from pathlib import Path

import numpy as np

from ..system import System


def read_fifo(fin, dtype, count):
    buffer = fin.read(int(dtype[-1]) * count)
    return np.frombuffer(buffer, dtype=dtype, count=count)


class IOFifo(object):
    def __init__(self, cwd=None):
        self.mode = "fifo"
        self.cwd = cwd

    def load_system(self, input, system=None, step=None):

        self._system = system

        if step is None:
            step = 0

        self._step = np.asarray(step)

        assert stat.S_ISFIFO(os.stat(input).st_mode)

        self._fin = open(input, "rb")

        self._n_atoms, self._n_qm_atoms, qm_charge, qm_mult, self._pbc = read_fifo(self._fin, dtype="i4", count=5)

        if system is None:
            self._system = System(self._n_atoms, self._n_qm_atoms, qm_charge=qm_charge, qm_mult=qm_mult)
            return self._system

    import line_profiler
    import atexit
    profile = line_profiler.LineProfiler()
    atexit.register(profile.print_stats)

    @profile
    def return_results(self, energy, forces, output=None):
        assert self._system is not None

        if output is None:
            output = Path(self._fin.name).with_suffix('.out')

        try:
            os.mkfifo(output)
        except:
            pass

        self._system.atoms.charges[:] = read_fifo(self._fin, dtype="f8", count=self._n_atoms)
        self._system.qm.atoms.elements[:] = read_fifo(self._fin, dtype="i4", count=self._n_qm_atoms)

        if self._pbc > 0:
            cell_basis = read_fifo(self._fin, dtype="f8", count=9).reshape(3, 3).copy()
            cell_basis[np.isclose(cell_basis, 0.0)] = 0.0
            self._system.cell_basis[:] = cell_basis

        # First cycle
        self._step[()] = read_fifo(self._fin, dtype="i4", count=1)

        if self._pbc == 2 :
            cell_basis = read_fifo(self._fin, dtype="f8", count=9).reshape(3, 3).copy()
            cell_basis[np.isclose(cell_basis, 0.0)] = 0.0
            self._system.cell_basis[:] = cell_basis

        self._system.atoms.positions[:] = read_fifo(self._fin, dtype="f8", count=self._n_atoms * 3).reshape(3, self._n_atoms)

        self._fout = open(output, "wb")
        self._fout.write(energy.tobytes())
        self._fout.write(forces.tobytes(order="F"))

        while True:

            try:
                _step = read_fifo(self._fin, dtype="i4", count=1)
                self._step[()] = _step
            except:
                break

            if self._pbc == 2 :
                cell_basis = read_fifo(self._fin, dtype="f8", count=9).reshape(3, 3).copy()
                cell_basis[np.isclose(cell_basis, 0.0)] = 0.0
                self._system.cell_basis[:] = cell_basis

            self._system.atoms.positions[:] = read_fifo(self._fin, dtype="f8", count=self._n_atoms * 3).reshape(3, self._n_atoms)

            _energy = energy.tobytes()
            self._fout.write(_energy)
            _forces = forces.tobytes(order="F")
            self._fout.write(_forces)