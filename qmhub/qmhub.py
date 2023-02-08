"""
QMHub
A universal QM/MM interface.
"""

from .simulation import Simulation
from .model import Model
from .engine import Engine
from .iotools import IO


class QMMM(object):
    '''QMHub python module'''
    def __init__(self, mode, driver=None, cwd=None):
        '''Initizalizes the QMHub module with the I/O mode, cwd, and driver.
        
        Args:
            mode (str):
            driver (str, optional):
            cwd (str, optional):
        '''
        self.io = IO.create(mode, cwd)
        self.driver = driver
        self.engine_groups = {}

    def setup_simulation(self, protocol="md", **kwargs):
        '''Prepares a simulations with given protocols.
        
        Args:
            protocol (str, optional): `md` is molecular dynamics.`mts` is multiple time step molecular dynamics, where multiple steps of molecular simulation are made between each quantum mechanical step. Default is `md`.
        '''
        self.simulation = Simulation(protocol, **kwargs)

    def load_system(self, input, save_input=False):
        ''''Loads a saved simulation from the step it was saved at. Takes the simualtion as input, which has saved it's last step.
        
        Args:
            input (str): The path to a binary file, text file, or a named pipe (fifo).
            save_input (Boolean, optional): If the files are to be kept after running.
        '''
        self.system = self.io.load_system(input, step=self.simulation.step)
        if save_input:
            self.io.save_input(input)

    def build_model(self, switching_type=None, cutoff=None, swdist=None, pbc=None):
        '''Creates a model to store atome positions and charges, takes several options with defaults of `None`.
        
        Args:
            switching_type (optional): 
            cutoff (optional): 
            swdit (optional):
            pbc (optional):
        '''
        if not hasattr(self, 'system'):
            raise AttributeError("Please load system first.")

        self.model = Model(
            self.system.qm.atoms.positions,
            self.system.atoms.positions,
            self.system.qm.atoms.charges,
            self.system.atoms.charges,
            self.system.cell_basis,
            self.system.qm_charge,
            switching_type=switching_type,
            cutoff=cutoff,
            swdist=swdist,
            pbc=pbc
        )

    def add_engine(self, engine, name=None, group_name=None, cwd=None, options=None):
        '''The engine to model atom positions and charges. Default engine name is `engine`.
        
        Args:
            engine ():
            name (str, optional): 
            group_name (str, optional): 
            cwd (str, optional):
            options (str, optional):
        '''
        name = name or engine
        group_name = group_name or "engine"
        cwd = cwd or self.io.cwd

        if not hasattr(self, group_name):
            group_obj = Engine(
                self.system.qm.atoms.positions,
                self.system.qm.atoms.elements,
                self.model.elec.embedding_mm_positions,
                self.model.elec.embedding_mm_charges,
                charge=self.system.qm_charge,
                mult=self.system.qm_mult,
            )
            setattr(self, group_name, group_obj)
            self.engine_groups[group_name] = group_obj

            self.model.get_result(
                name=group_name,
                qm_energy=group_obj.qm_energy,
                qm_energy_gradient=group_obj.qm_energy_gradient,
                mm_esp=group_obj.mm_esp,
            )

            result_obj = getattr(self.model, group_name)
            self.simulation.add_engine(group_name, result_obj)

        group_obj = self.engine_groups[group_name]
        group_obj.add_engine(engine, name=name, cwd=cwd, options=options)

    def return_results(self, output=None):
        '''Prints the simulation energy figures and energy gradient base on current simulations.
        
        Args:
            output (str, optional): '''
        self.io.return_results(self.simulation.energy, self.simulation.energy_gradient, output)
