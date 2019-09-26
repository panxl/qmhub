import os
import subprocess as sp
import numpy as np


class QMBase(object):

    QMTOOL = None

    def __init__(self):
        raise NotImplementedError()

    def gen_input(self):
        raise NotImplementedError()

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        raise NotImplementedError()

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        raise NotImplementedError()

    def run(self, read_guess=False):
        """Run QM calculation."""

        cmdline = self.gen_cmdline()

        if not read_guess:
            self.rm_guess()

        proc = sp.Popen(args=cmdline, shell=True)
        proc.wait()
        self.exitcode = proc.returncode
        return self.exitcode

    @staticmethod
    def get_nproc():
        """Get the number of processes for QM calculation."""
        if "OMP_NUM_THREADS" in os.environ:
            nproc = int(os.environ["OMP_NUM_THREADS"])
        elif "SLURM_NTASKS" in os.environ:
            nproc = int(os.environ["SLURM_NTASKS"])
        else:
            nproc = 1
        return nproc

    @staticmethod
    def load_output(output_file):
        """Load output file."""

        f = open(output_file, "r")
        output = f.readlines()
        f.close()

        return output
