import os
import shutil
import numpy as np

from .. import units

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class ORCA(QMBase):

    QMTOOL = 'ORCA'

    def get_qm_params(self, method=None, basis=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(ORCA, self).get_qm_params(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for ORCA.")

        if basis is not None:
            self.basis = basis
        else:
            raise ValueError("Please set basis for ORCA.")

    def gen_input(self, path=None):
        """Generate input file for QM software."""

        qmtmpl = QMTmpl(self.QMTOOL)

        if self.calc_forces:
            calc_forces = 'EnGrad '
        else:
            calc_forces = ''

        if self.read_guess:
            read_guess = ''
        else:
            read_guess = 'NoAutoStart '

        if self.addparam is not None:
            if isinstance(self.addparam, list):
                addparam = "".join(["%s " % i for i in self.addparam])
            else:
                addparam = self.addparam + " "
        else:
            addparam = ''

        nproc = self.get_nproc()

        if path is None:
            path = self.basedir

        with open(os.path.join(path, "orca.inp"), 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(
                method=self.method, basis=self.basis,
                calc_forces=calc_forces, read_guess=read_guess,
                addparam=addparam, nproc=nproc,
                pntchrgspath="\"orca.pntchrg\""))
            f.write("%coords\n")
            f.write("  CTyp xyz\n")
            f.write("  Charge %d\n" % self.charge)
            f.write("  Mult %d\n" % self.mult)
            f.write("  Units Angs\n")
            f.write("  coords\n")

            for i in range(self._n_qm_atoms):
                f.write(" ".join(["%6s" % self._qm_element[i],
                                  "%22.14e" % self._qm_position[i, 0],
                                  "%22.14e" % self._qm_position[i, 1],
                                  "%22.14e" % self._qm_position[i, 2], "\n"]))
            f.write("  end\n")
            f.write("end\n")

        with open(os.path.join(path, "orca.pntchrg"), 'w') as f:
            f.write("%d\n" % self._n_mm_atoms)
            for i in range(self._n_mm_atoms):
                f.write("".join(["%22.14e " % self._mm_charge[i],
                                 "%22.14e" % self._mm_position[i, 0],
                                 "%22.14e" % self._mm_position[i, 1],
                                 "%22.14e" % self._mm_position[i, 2], "\n"]))

        with open(os.path.join(path, "orca.pntvpot.xyz"), 'w') as f:
            f.write("%d\n" % self._n_mm_atoms)
            for i in range(self._n_mm_atoms):
                f.write("".join(["%22.14e" % (self._mm_position[i, 0] / units.L_AU),
                                 "%22.14e" % (self._mm_position[i, 1] / units.L_AU),
                                 "%22.14e" % (self._mm_position[i, 2] / units.L_AU), "\n"]))

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