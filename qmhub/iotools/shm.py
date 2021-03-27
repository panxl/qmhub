import os, stat, time
from pathlib import Path
from mmap import mmap

import numpy as np
import posix_ipc

from ..system import System


def read_fifo(fin, dtype, count):
    buffer = fin.read(int(dtype[-1]) * count)
    return np.frombuffer(buffer, dtype=dtype, count=count)


class IOShm(object):
    def __init__(self, cwd=None):
        self.mode = "shm"
        self.cwd = cwd

    def load_system(self, input, system=None, step=None):

        self.cwd = self.cwd or Path(input).parent

        if step is None:
            step = 0

        self._step = np.asarray(step)

        assert stat.S_ISFIFO(os.stat(input).st_mode)

        self._fin = open(input, "rb")

        self._n_atoms, self._n_qm_atoms, qm_charge, qm_mult, self._pbc = read_fifo(self._fin, dtype="i4", count=5)
        self._system = system or System(self._n_atoms, self._n_qm_atoms, qm_charge=qm_charge, qm_mult=qm_mult)
        self._n_mm_atoms = self._n_atoms - self._n_qm_atoms

        return self._system

    def return_results(self, energy, forces, output=None):
        assert self._system is not None

        output = output or Path(self._fin.name).with_suffix('.out')

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

        shm = posix_ipc.SharedMemory("/qmmm")
        shm_buf = mmap(shm.fd, shm.size)
        shm.close_fd()

        self._system.qm.atoms.positions[:] = np.ndarray(shape=(3, self._n_qm_atoms), dtype='float64', buffer=shm_buf, order="F")

        if self._n_mm_atoms:
            self._system.mm.atoms.positions[:] = np.ndarray(shape=(4, self._n_mm_atoms), dtype='float64', buffer=shm_buf,
                    offset=3*self._n_qm_atoms*np.float_().itemsize, order="F")[:3]

        _forces = forces.tobytes(order="F")
        shm_buf[:len(_forces)] = _forces

        self._fout = open(output, "wb")
        self._fout.write(energy.tobytes())
        self._fout.flush()

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

            self._system.qm.atoms.positions[:] = np.ndarray(shape=(3, self._n_qm_atoms), dtype='float64', buffer=shm_buf, order="F")

            if self._n_mm_atoms:
                self._system.mm.atoms.positions[:] = np.ndarray(shape=(4, self._n_mm_atoms), dtype='float64', buffer=shm_buf,
                        offset=3*self._n_qm_atoms*np.float_().itemsize, order="F")[:3]

            _forces = forces.tobytes(order="F")
            shm_buf[:len(_forces)] = _forces

            self._fout.write(energy.tobytes())
            self._fout.flush()

    @staticmethod
    def save_input(input):
        """Preserve the input file passed from the driver."""
        pass
