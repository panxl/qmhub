import os
import subprocess as sp


def run_cmdline(cmdline):
    """Run QM calculation."""

    proc = sp.Popen(args=cmdline, shell=True)
    proc.wait()
    return proc.returncode


def get_nproc():
    """Get the number of processes for QM calculation."""
    if "OMP_NUM_THREADS" in os.environ:
        nproc = int(os.environ["OMP_NUM_THREADS"])
    elif "SLURM_NTASKS" in os.environ:
        nproc = int(os.environ["SLURM_NTASKS"])
    else:
        nproc = 1
    return nproc
