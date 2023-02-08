import numpy as np


from .utils.darray import DependArray


class Simulation(object):
    '''Simulation object stores energy gradient and energy values as an array with a dependency on the simulation step; among other factors such as the engine, protocol, and scaling factor needed by the simulation.'''
    def __init__(self, protocol=None, engine_name=None, engine2_name=None, *, nrespa=None, scaling_factor=None):
        '''
        Initizalizes the simulation object by storing the variable the array needs, placing them in a list as needed to form the energy array and energy gradient array.
        
        Args:
            protocol (str, optional):
            engine_name (str, optional):
            engine2_name (str, optional):
            * [what]
            nrespa (int, optional):
            scaling_factor(int, optional):
        '''

        self.protocol = protocol or "md"

        self.engine_name = engine_name or "engine"

        if self.protocol.lower() == "mts":
            engine2_name = engine2_name or "engine2"
        self.engine2_name = engine2_name

        self.nrespa = nrespa or 1
        self.scaling_factor = scaling_factor

        self.step = DependArray(np.array(0), name="step")

        self.energy = DependArray(
            name="energy",
            func=Simulation._get_energy,
            kwargs={
                'protocol': self.protocol,
                'nrespa': self.nrespa,
                'scaling_factor': self.scaling_factor,
            },
            dependencies=[self.step],
        )

        self.energy_gradient = DependArray(
            name="energy_gradient",
            func=Simulation._get_energy_gradient,
            kwargs={
                'protocol': self.protocol,
                'nrespa': self.nrespa,
                'scaling_factor': self.scaling_factor,
            },
            dependencies=[self.step],
        )

    def add_engine(self, name, engine):
        '''
        Attaches engine to simulation object, and gives the energy array and energy gradient arrays those arrays brought by the engine.
        
        Args:
            name (str):
            engine (str):
        '''
        if name == self.engine2_name and not hasattr(self, self.engine_name):
            raise ValueError("Please add engine before adding engine2.")

        if name in [self.engine_name, self.engine2_name]:
            setattr(self, name, engine)
        else:
            raise ValueError(f"Please add {self.engine_name} or {self.engine2_name}.")

        self.energy.add_dependency(engine.energy)
        self.energy_gradient.add_dependency(engine.energy_gradient)

    @staticmethod
    def _get_energy(step, energy, energy2=None, protocol=None, nrespa=None, scaling_factor=None):
        '''Retuerns the free energy of the system as a float. A static method for all instances of the simulation class. 
        Will return 0 if `((step + 1) % nrespa)`
        
        Args:
            step (int):
            energy (Array):
            energy2 (Array, optional):
            protocol (str, optional):
            nrespa (int, optional):
            scaling_factor (int, optional):
        Returns:
            float
        '''
        if scaling_factor is not None:
            energy = energy * scaling_factor
        if protocol.lower() == "md":
            return energy
        elif protocol.lower() == "mts":
            if (step < nrespa):
                return energy
            elif ((step + 1) % nrespa) == 0:
                return energy
            else:
                return 0.
        else:
            raise ValueError("Only 'md' and 'mts' are supported.")

    @staticmethod
    def _get_energy_gradient(step, gradient, gradient2=None, protocol=None, nrespa=None, scaling_factor=None):
        '''Returns the energy gradient and as Numpy array output. A static method for all instances of the simulation class. 
        
        Args:
           step ():
           gradient ():
           gradient2 ( optional):
        Returns:
            Numpy Array
        '''
        if scaling_factor is not None:
            gradient = gradient * scaling_factor
            if gradient2 is not None:
                gradient2 = gradient2 * scaling_factor
        if protocol.lower() == "md":
            return gradient
        elif protocol.lower() == "mts":
            if (step < nrespa):
                return gradient
            elif ((step + 1) % nrespa) == 0:
                return gradient2 + nrespa * (gradient - gradient2)
            else:
                return gradient2
        else:
            raise ValueError("Only 'md' and 'mts' are supported.")
