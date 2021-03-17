import os
from copy import copy
from pathlib import Path

import numpy as np

from ..utils.dobject import invalidate_cache
from ..utils.darray import DependArray
from ..utils.dlist import DependList
from ..utils.elements import get_element_symbols
from ..utils.sys import run_cmdline, get_nproc


class QMBase(object):

    OUTPUT = None
    default_options = None

    def __init__(
        self,
        qm_positions,
        qm_elements,
        mm_positions,
        mm_charges,
        charge=None,
        mult=None,
        cwd=None,
        options=None,
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

        self.cwd = cwd or os.getcwd()
        self.nproc = get_nproc()
        self.cmdline = self.gen_cmdline()

        self.qm_element_symbols = DependArray(
            name="qm_element_symbols",
            func=get_element_symbols,
            dependencies=[
                self.qm_elements,
            ]
        )
        self.qm_element_ids = DependArray(
            name="qm_element_ids",
            func=(lambda t : [np.unique(t).tolist().index(i) for i in t]),
            dependencies=[
                self.qm_elements,
            ]
        )
        self._qm_cache = DependList(
            name="qm_cache",
            func=self._get_qm_cache,
            kwargs={"output": self.OUTPUT},
            dependencies=[
                self.qm_positions,
                self.qm_elements,
                self.mm_positions,
                self.mm_charges,
            ]
        )

        self.options = copy(self.default_options)
        self.update_options(options)

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

    def _get_qm_cache(self, *args, output=None):
        self.gen_input()
        run_cmdline(self.cmdline)
        if output is not None:
            output_path = Path(self.cwd).joinpath(output)
            try:
                output = output_path.read_text().split("\n")
            except:
                raise
            else:
                os.remove(output_path)
        return output or []

    def update_options(self, options=None):
        if options is not None:
            for key, value in options.items():
                try:
                    self.options[key] = type(self.options[key])(value)
                except:
                    self.options[key] = value

        invalidate_cache(self._qm_cache)

    def gen_input(self):
        """Generate input file for QM software."""

        raise NotImplementedError()

    def gen_cmdline(self):
        """Generate commandline for QM calculation."""
        pass

    def _get_qm_energy(self, qm_cache=None):
        """Get QM energy from output of QM calculation."""

        raise NotImplementedError()

    def _get_qm_energy_gradient(self, qm_cache=None):
        """Get QM energy gradient from output of QM calculation."""

        raise NotImplementedError()

    def _get_mm_esp(self, qm_cache=None):
        """Get electrostatic potential at MM atoms in the near field from QM density."""

        return np.zeros((4, len(self.mm_charges)))

    def _get_mulliken_charges(self, qm_cache=None):
        """Get Mulliken charges from output of QM calculation."""

        raise NotImplementedError()
