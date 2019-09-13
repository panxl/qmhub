import os
import subprocess as sp
import numpy as np


class QMBase(object):

    QMTOOL = None

    def __init__(self, basedir, charge=None, mult=None):
        """
        Creat a QM object.
        """

        self.basedir = basedir

        if charge is not None:
            self.charge = charge
        else:
            raise ValueError("Please set 'charge' for QM calculation.")
        if mult is not None:
            self.mult = mult
        else:
            self.mult = 1

    def set_qm_atoms(self, coordinates, elements):
        self.qm_coordinates = coordinates
        self.qm_elements = elements

    def set_external_charges(self, coordinates, charges):
        self.extchg_coordinates = coordinates
        self.extchg_charges = charges

    @staticmethod
    def get_nproc():
        """Get the number of processes for QM calculation."""
        if 'OMP_NUM_THREADS' in os.environ:
            nproc = int(os.environ['OMP_NUM_THREADS'])
        elif 'SLURM_NTASKS' in os.environ:
            nproc = int(os.environ['SLURM_NTASKS']) - 4
        else:
            nproc = 1
        return nproc

    @staticmethod
    def load_output(output_file):
        """Load output file."""

        f = open(output_file, 'r')
        output = f.readlines()
        f.close()

        return output

    def get_qm_params(self, calc_forces=None, read_guess=None, addparam=None):
        if calc_forces is not None:
            self.calc_forces = calc_forces
        elif not hasattr(self, 'calc_forces'):
            self.calc_forces = True

        if read_guess is not None:
            self.read_guess = read_guess
        elif not hasattr(self, 'read_guess'):
            self.read_guess = False

        self.addparam = addparam

    def run(self):
        """Run QM calculation."""

        cmdline = self.gen_cmdline()

        if not self.read_guess:
            self.rm_guess()

        proc = sp.Popen(args=cmdline, shell=True)
        proc.wait()
        self.exitcode = proc.returncode
        return self.exitcode

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        raise NotImplementedError()

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        raise NotImplementedError()