import os
from pathlib import Path
import shutil
import numpy as np

from .qmbase import QMBase
from ..qmtmpl import QMTmpl


class QChem(QMBase):

    QMTOOL = "Q-Chem"

    def __init__(
        self,
        qm_positions,
        qm_elements,
        mm_positions=None,
        mm_charges=None,
        charge=None,
        mult=None,
        ):
        """
        Creat a QM object.
        """

        self.qm_positions = qm_positions
        self.qm_elements = qm_elements
        self.mm_positions = mm_positions
        self.mm_charges = mm_charges

        if charge is not None:
            self.charge = charge
        else:
            raise ValueError("Please set 'charge' for QM calculation.")
        if mult is not None:
            self.mult = mult
        else:
            self.mult = 1

    def gen_input(
        self,
        method=None,
        basis=None,
        calc_forces=None,
        read_guess=None,
        basedir=None,
    ):
        """Generate input file for QM software."""

        qmtmpl = QMTmpl(self.QMTOOL)

        if method is None:
            raise ValueError("Please set method for Q-Chem.")

        if basis is None:
            raise ValueError("Please set basis for Q-Chem.")

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

        if basedir is not None:
            basedir = basedir
        else:
            basedir = os.getcwd()

        with open(os.path.join(basedir, "qchem.inp"), "w") as f:
            f.write(
                qmtmpl.gen_qmtmpl().substitute(
                    jobtype=jobtype,
                    method=method,
                    basis=basis,
                    read_guess=read_guess,
                    qm_mm=qm_mm,
                )
            )
            f.write("$molecule\n")
            f.write("%d %d\n" % (self.charge, self.mult))

            qm_elements = np.asarray(self.qm_elements, dtype=self.qm_elements.dtype)
            qm_positions = np.asarray(self.qm_positions, dtype=self.qm_positions.dtype)

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
            if self.mm_charges is not None:
                mm_charges = np.asarray(self.mm_charges, dtype=self.mm_charges.dtype)
                mm_positions = np.asarray(self.mm_positions, dtype=self.mm_positions.dtype)
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

    def gen_cmdline(self, basedir=None):
        """Generate commandline for QM calculation."""

        if basedir is not None:
            basedir = basedir
        else:
            basedir = os.getcwd()

        nproc = self.get_nproc()
        cmdline = "cd " + basedir + "; "
        cmdline += "qchem -nt %d qchem.inp qchem.out save > qchem_run.log" % nproc

        return cmdline

    def rm_guess(self):
        """Remove save from previous QM calculation."""

        if "QCSCRATCH" in os.environ:
            qmsave = os.environ["QCSCRATCH"] + "/save"
            if os.path.isdir(qmsave):
                shutil.rmtree(qmsave)

    def parse_output(self, basedir=None, calc_forces=True):
        """Parse the output of QM calculation."""

        if basedir is not None:
            basedir = basedir
        else:
            basedir = os.getcwd()

        output = Path(basedir).joinpath("qchem.out").read_text().split("\n")
    
        self.qm_energy = self.get_qm_energy(output)

        if calc_forces:
            output = Path(basedir).joinpath("efield.dat").read_text().split("\n")
            self.qm_forces = self.get_qm_forces(output)
            self.mm_efield = self.get_mm_efield(output)

            output = Path(basedir).joinpath("esp.dat").read_text().split("\n")
            self.mm_esp = self.get_mm_esp(output)

    def get_qm_energy(self, output):
        """Get QM energy from output of QM calculation."""

        if isinstance(output, str):
            try:
                output = Path(output).read_text().split("\n")
            except:
                output = output.split("\n")

        for line in output:
            line = line.strip().expandtabs()

            cc_energy = 0.0
            if "Charge-charge energy" in line:
                cc_energy = line.split()[-2]

            if "Total energy" in line:
                scf_energy = line.split()[-1]
                break

        return float(scf_energy) - float(cc_energy)

    def get_qm_forces(self, output):
        """Get QM forces from output of QM calculation."""

        if isinstance(output, str):
            try:
                output = Path(output).read_text().split("\n")
            except:
                output = output.split("\n")

        return np.loadtxt(output[len(self.mm_charges):], dtype=float)

    def get_mm_efield(self, output):
        """Get QM forces from output of QM calculation."""

        if isinstance(output, str):
            try:
                output = Path(output).read_text().split("\n")
            except:
                output = output.split("\n")

        return np.loadtxt(output[:len(self.mm_charges)], dtype=float)

    def get_mm_esp(self, output):
        """Get ESP at MM atoms in the near field from QM density."""

        if isinstance(output, str):
            try:
                output = Path(output).read_text().split("\n")
            except:
                output = output.split("\n")

        return np.loadtxt(output)

    def get_mulliken_charges(self, output):
        """Get Mulliken charges from output of QM calculation."""

        if isinstance(output, str):
            try:
                output = Path(output).read_text().split("\n")
            except:
                output = output.split("\n")

        charge_string = "Ground-State Mulliken Net Atomic Charges"

        for i in range(len(output)):
            if charge_string in output[i]:
                mulliken_charges = np.empty(len(self.qm_elements), dtype=float)
                for j in range(len(self.qm_elements)):
                    line = output[i + 4 + j]
                    mulliken_charges[j] = float(line.split()[2])
                break

        return mulliken_charges
