from __future__ import division

import os
import shutil
import numpy as np

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class QChem(QMBase):

    QMTOOL = 'Q-Chem'

    def get_qm_params(self, method=None, basis=None, **kwargs):
        """Get the parameters for QM calculation."""

        super(QChem, self).get_qm_params(**kwargs)

        if method is not None:
            self.method = method
        else:
            raise ValueError("Please set method for Q-Chem.")

        if basis is not None:
            self.basis = basis
        else:
            raise ValueError("Please set basis for Q-Chem.")

    def gen_input(self, path=None):
        """Generate input file for QM software."""

        qmtmpl = QMTmpl(self.QMTOOL)

        if self.calc_forces:
            jobtype = 'force'
            qm_mm = 'true'
        else:
            jobtype = 'sp'
            qm_mm = 'false'

        if self.read_guess:
            read_guess = 'scf_guess read\n'
        else:
            read_guess = ''

        if self.addparam is not None:
            if isinstance(self.addparam, list):
                addparam = "".join(["%s\n" % i for i in self.addparam])
            else:
                addparam = self.addparam + '\n'
        else:
            addparam = ''

        if path is None:
            path = self.basedir

        with open(os.path.join(path, "qchem.inp"), 'w') as f:
            f.write(qmtmpl.gen_qmtmpl().substitute(
                jobtype=jobtype, method=self.method, basis=self.basis,
                read_guess=read_guess, qm_mm=qm_mm, esp_and_asp=esp_and_asp,
                addparam=addparam))
            f.write("$molecule\n")
            f.write("%d %d\n" % (self.charge, self.mult))

            for i in range(self._n_qm_atoms):
                f.write("".join(["%3s" % self._qm_element[i],
                                 "%22.14e" % self._qm_position[i, 0],
                                 "%22.14e" % self._qm_position[i, 1],
                                 "%22.14e" % self._qm_position[i, 2], "\n"]))
            f.write("$end" + "\n\n")

            f.write("$external_charges\n")
            if self._mm_charge is not None:
                for i in range(self._n_mm_atoms):
                    f.write("".join(["%22.14e" % self._mm_position[i, 0],
                                     "%22.14e" % self._mm_position[i, 1],
                                     "%22.14e" % self._mm_position[i, 2],
                                     " %22.14e" % self._mm_charge[i], "\n"]))
            f.write("$end" + "\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        nproc = self.get_nproc()
        cmdline = "cd " + self.basedir + "; "
        cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        if 'QCSCRATCH' in os.environ:
            qmsave = os.environ['QCSCRATCH'] + "/save"
            if os.path.isdir(qmsave):
                shutil.rmtree(qmsave)

    def parse_output(self):
        """Parse the output of QM calculation."""

        output = self.load_output(os.path.join(self.basedir, "qchem.out"))

        self.get_qm_energy(output)
        self.get_qm_charge(output)

        output = self.load_output(os.path.join(self.basedir, "efield.dat"))
        self.get_qm_force(output)

    def get_qm_energy(self, output=None):
        """Get QM energy from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "qchem.out"))

        for line in output:
            line = line.strip().expandtabs()

            cc_energy = 0.0
            if "Charge-charge energy" in line:
                cc_energy = line.split()[-2]

            if "Total energy" in line:
                scf_energy = line.split()[-1]
                break

        self.qm_energy = float(scf_energy) - float(cc_energy)

        return self.qm_energy

    def get_qm_charge(self, output=None):
        """Get Mulliken charges from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "qchem.out"))

        if self._qm_esp is not None:
            charge_string = "Merz-Kollman ESP Net Atomic Charges"
        else:
            charge_string = "Ground-State Mulliken Net Atomic Charges"

        for i in range(len(output)):
            if charge_string in output[i]:
                self.qm_charge = np.empty(self._n_qm_atoms, dtype=float)
                for j in range(self._n_qm_atoms):
                    line = output[i + 4 + j]
                    self.qm_charge[j] = float(line.split()[2])
                break

        return self.qm_charge

    def get_qm_force(self, output=None):
        """Get QM forces from output of QM calculation."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "efield.dat"))

        self.qm_force = -1 * np.loadtxt(output[(-1 * self._n_qm_atoms):], dtype=float)

        return self.qm_force

    def get_mm_esp_eed(self, output=None):
        """Get ESP at MM atoms in the near field from QM density."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "esp.dat"))

        self.mm_esp_eed = np.loadtxt(output)

        return self.mm_esp_eed