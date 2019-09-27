import os
import shutil
import numpy as np

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class QChem(QMBase):

    QMTOOL = "Q-Chem"

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
            raise ValueError("Please set method for Q-Chem.")

        if basis is not None:
            self.basis = basis
        else:
            raise ValueError("Please set basis for Q-Chem.")

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
            jobtype = "force"
            qm_mm = "true"
        else:
            jobtype = "sp"
            qm_mm = "false"

        if read_guess:
            read_guess = "scf_guess read\n"
        else:
            read_guess = ""

        if path is None:
            path = self.basedir

        with open(os.path.join(path, "qchem.inp"), "w") as f:
            f.write(
                qmtmpl.gen_qmtmpl().substitute(
                    jobtype=jobtype,
                    method=self.method,
                    basis=self.basis,
                    read_guess=read_guess,
                    qm_mm=qm_mm,
                )
            )
            f.write("$molecule\n")
            f.write("%d %d\n" % (charge, mult))

            for i in range(len(qm_elements)):
                f.write(
                    "".join(
                        [
                            "%3s" % qm_elements[i],
                            "%22.14e" % qm_positions[0, i],
                            "%22.14e" % qm_positions[1, i],
                            "%22.14e" % qm_positions[2, i],
                            "\n",
                        ]
                    )
                )
            f.write("$end" + "\n\n")

            f.write("$external_charges\n")
            if mm_charges is not None:
                for i in range(len(mm_charges)):
                    f.write(
                        "".join(
                            [
                                "%22.14e" % mm_positions[0, i],
                                "%22.14e" % mm_positions[1, i],
                                "%22.14e" % mm_positions[2, i],
                                " %22.14e" % mm_charges[i],
                                "\n",
                            ]
                        )
                    )
            f.write("$end" + "\n")

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        nproc = self.get_nproc()
        cmdline = "cd " + self.basedir + "; "
        cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        if "QCSCRATCH" in os.environ:
            qmsave = os.environ["QCSCRATCH"] + "/save"
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

        self.qm_force = -1 * np.loadtxt(output[(-1 * self._n_qm_atoms) :], dtype=float)

        return self.qm_force

    def get_mm_esp_eed(self, output=None):
        """Get ESP at MM atoms in the near field from QM density."""

        if output is None:
            output = self.load_output(os.path.join(self.basedir, "esp.dat"))

        self.mm_esp_eed = np.loadtxt(output)

        return self.mm_esp_eed
