import os
import numpy as np

from ..utils import DependArray, invalidate_cache, run_cmdline


class QMBase(object):

    QMTOOL = None

    def __init__(
        self,
        qm_positions,
        qm_elements,
        mm_positions,
        mm_charges,
        charge=None,
        mult=None,
        basedir=None,
        keywords=None,
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

        if basedir is not None:
            self.basedir = basedir
        else:
            self.basedir = os.getcwd()

        if keywords is not None:
            self.keywords = keywords
        else:
            self.keywords = {}

        self._qm_cache = DependArray(
            name="qm_updated",
            func=self._get_qm_cache,
            dependencies=[
                self.qm_positions,
                self.qm_elements,
                self.mm_positions,
                self.mm_charges,
            ]
        )
        self.qm_energy = DependArray(
            name="qm_energy",
            func=self._get_qm_energy,
            dependencies=[self._qm_cache],
        )
        self.qm_energy_gradient = DependArray(
            name="qm_energy_gradient",
            func=self._get_qm_energy_gradient,
            dependencies=[self._qm_cache],
        )
        self.mm_esp = DependArray(
            name="mm_esp",
            func=self._get_mm_esp,
            dependencies=[self._qm_cache],
        )
        self.mulliken_charges = DependArray(
            name="mulliken_charges",
            func=self._get_mulliken_charges,
            dependencies=[self._qm_cache],
        )

    def _get_qm_cache(self, *args):
        self.gen_input()
        run_cmdline(self.gen_cmdline())

        return True

    def update_keywords(self, keywords):
        self.keywords.update(keywords)
        invalidate_cache(self._qm_cache)

    def gen_input(self):
        """Generate input file for QM software."""

        raise NotImplementedError()

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""

        raise NotImplementedError()

    def _get_qm_energy(self, qm_cache=None):
        """Get QM energy from output of QM calculation."""

        raise NotImplementedError()

    def _get_qm_energy_gradient(self, qm_cache=None):
        """Get QM energy gradient from output of QM calculation."""

        raise NotImplementedError()

    def _get_mm_esp(self, qm_cache=None):
        """Get electrostatic potential at MM atoms in the near field from QM density."""

        raise NotImplementedError()

    def _get_mulliken_charges(self, qm_cache=None):
        """Get Mulliken charges from output of QM calculation."""

        raise NotImplementedError()
