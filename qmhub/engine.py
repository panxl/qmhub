import os
import subprocess as sp
import numpy as np

from .qmtools import QM
from .utils.darray import DependArray

class Engine(object):
    '''QMHub Engine object'''
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
        Creates an Engine object for QM simulations; which stores the elements and positions of the QM simulation, and optionally the positions and charges from the MM simulation. Also creates energy arrays and energy gradients arrays particular to the QM system.
        
        Args:
            qm_positions ():
            qm_elements ():
            mm_positions (optional):
            mm_charges (optional):
            charge (optional):
            mult (optional):
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

        self.engines = {}

        self.qm_energy = DependArray(
            name="qm_energy",
            func=(lambda *args: sum(args)),
            dependencies=[],
        )
        self.qm_energy_gradient = DependArray(
            name="qm_energy_gradient",
            func=(lambda *args: sum(args)),
            dependencies=[],
        )
        self.mm_esp = DependArray(
            name="mm_esp",
            func=(lambda *args: sum(args)),
            dependencies=[],
        )

    def add_engine(self, engine, name=None, cwd=None, options=None):
        '''
        Attaches an given engine to the dictionary of the Engine object, i.e. the set of engines; allowing for the results of different engines to be added together for additional module components, e.g. error correction.
        
        Args:
            engine (Engine):
            name (str, optional):
            cwd (optional):
            options (str, optional):
        '''
        name = name or engine

        engine_obj = QM.create(
            engine.lower(),
            qm_positions = self.qm_positions,
            qm_elements = self.qm_elements,
            mm_positions = self.mm_positions,
            mm_charges = self.mm_charges,
            charge = self.charge,
            mult = self.mult,
            cwd = cwd,
            options = options,
        )

        setattr(self, name, engine_obj)
        self.engines[name] = engine_obj

        self.qm_energy.add_dependency(engine_obj.qm_energy)
        self.qm_energy_gradient.add_dependency(engine_obj.qm_energy_gradient)
        self.mm_esp.add_dependency(engine_obj.mm_esp)
