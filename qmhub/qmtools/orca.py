import os
import shutil
import numpy as np

from ..units import BOHR_IN_ANGSTROM

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class ORCA(QMBase):

    QMTOOL = 'ORCA'

    def __init__(self, basedir=None, method=None, basis=None):
        """
        Creat a QM object.
        """

        if basedir is not None:
            self.basedir = basedir
        else:
            self.basedir = os.getcwd()

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for ORCA.")

        if basis is not None:
            self.basis = basis
        else:
            raise ValueError("Please set basis for ORCA.")

    def gen_input(
        self,
        qm_positions,
        qm_elements,
        mm_positions=None,
        mm_charges=None,
        charge=None,
        mult=None,
        calc_forces=None,
        read_guess=None,
        path=None,
    ):
        """Generate input file for QM software."""

        qmtmpl = QMTmpl(self.QMTOOL)

        if calc_forces:
            calc_forces = 'EnGrad '
        else:
            calc_forces = ''

        if read_guess:
            read_guess = ''
        else:
            read_guess = 'NoAutoStart '

        nproc = self.get_nproc()

        if path is None:
            path = self.basedir

        with open(os.path.join(path, "orca.inp"), 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(
                method=self.method, basis=self.basis,
                calc_forces=calc_forces, read_guess=read_guess,
                nproc=nproc, pntchrgspath="\"orca.pntchrg\""))
            f.write("%coords\n")
            f.write("  CTyp xyz\n")
            f.write("  Charge %d\n" % charge)
            f.write("  Mult %d\n" % mult)
            f.write("  Units Angs\n")
            f.write("  coords\n")

            for i in range(len(qm_elements)):
                f.write(" ".join(["%6s" % qm_elements[i],
                                  "%22.14e" % qm_positions[0, i],
                                  "%22.14e" % qm_positions[1, i],
                                  "%22.14e" % qm_positions[2, i], "\n"]))
            f.write("  end\n")
            f.write("end\n")

        with open(os.path.join(path, "orca.pntchrg"), 'w') as f:
            f.write("%d\n" % len(mm_charges))
            for i in range(len(mm_charges)):
                f.write("".join(["%22.14e " % mm_charges[i],
                                 "%22.14e" % mm_positions[0, i],
                                 "%22.14e" % mm_positions[1, i],
                                 "%22.14e" % mm_positions[2, i], "\n"]))

        with open(os.path.join(path, "orca.pntvpot.xyz"), 'w') as f:
            f.write("%d\n" % len(mm_charges))
            for i in range(len(mm_charges)):
                f.write("".join(["%22.14e" % (mm_positions[0, i] / BOHR_IN_ANGSTROM),
                                 "%22.14e" % (mm_positions[1, i] / BOHR_IN_ANGSTROM),
                                 "%22.14e" % (mm_positions[2, i] / BOHR_IN_ANGSTROM), "\n"]))

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        cmdline = "cd " + self.basedir + "; "
        cmdline += shutil.which("orca") + " orca.inp > orca.out; "
        cmdline += shutil.which("orca_vpot") + " orca.gbw orca.scfp orca.pntvpot.xyz orca.pntvpot.out >> orca.out"

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        qmsave = os.path.join(self.basedir, "orca.gbw")
        if os.path.isfile(qmsave):
            os.remove(qmsave)

    def parse_output(self):
        """Parse the output of QM calculation."""

        output = self.load_output(os.path.join(self.basedir, "orca.out"))

        self.get_qm_energy(output)
        self.get_qm_charge(output)
        self.get_qm_force()
        self.get_mm_force()

        self.get_mm_esp_eed()

    def get_qm_energy(self, output=None):
        """Get QM energy from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "orca.out"))

        for line in output:
            line = line.strip().expandtabs()

            if "FINAL SINGLE POINT ENERGY" in line:
                self.qm_energy = float(line.split()[-1])
                break

        return self.qm_energy

    def get_qm_charge(self, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "orca.out"))

        for i in range(len(output)):
            if "MULLIKEN ATOMIC CHARGES" in output[i]:
                charges = []
                for line in output[(i + 2):(i + 2 + self._n_qm_atoms)]:
                    charges.append(float(line.split()[3]))
                break

        self.qm_charge = np.array(charges)

        return self.qm_charge

    def get_qm_force(self, output=None):
        """Get QM forces from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "orca.engrad"))

        start = 11
        stop = start + self._n_qm_atoms * 3
        self.qm_force = -1 * np.loadtxt(output[start:stop]).reshape((self._n_qm_atoms, 3))

        return self.qm_force

    def get_mm_force(self, output=None):
        """Get external point charge forces from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "orca.pcgrad"))

        self.mm_force = -1 * np.loadtxt(output[1:(self._n_mm_atoms + 1)])

        return self.mm_force

    def get_mm_esp_eed(self, output=None):
        """Get ESP at MM atoms in the near field from QM density."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "orca.pntvpot.out"))

        self.mm_esp_eed = np.loadtxt(output[1:(self._n_mm_atoms + 1)], usecols=3)

        return self.mm_esp_eed