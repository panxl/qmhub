import numpy as np


from .utils.darray import DependArray


class Simulation(object):
    def __init__(self, protocol=None, engine_name=None, engine2_name=None, *, nrespa=None):

        self.protocol = protocol or "md"

        self.engine_name = engine_name or "engine"

        if self.protocol.lower() == "mts":
            engine2_name = engine2_name or "engine2"
        self.engine2_name = engine2_name

        self.nrespa = nrespa or 1

        self.step = DependArray(np.array(0), name="step")

        self.energy = DependArray(
            name="energy",
            func=Simulation._get_energy,
            kwargs={
                'protocol': self.protocol,
                'nrespa': self.nrespa,
            },
            dependencies=[self.step],
        )

        self.energy_gradient = DependArray(
            name="energy_gradient",
            func=Simulation._get_energy_gradient,
            kwargs={
                'protocol': self.protocol,
                'nrespa': self.nrespa,
            },
            dependencies=[self.step],
        )

    def add_engine(self, name, engine):
        if name == self.engine2_name and not hasattr(self, self.engine_name):
            raise ValueError("Please add engine before adding engine2.")

        if name in [self.engine_name, self.engine2_name]:
            setattr(self, name, engine)
        else:
            raise ValueError(f"Please add {self.engine_name} or {self.engine2_name}.")

        self.energy.add_dependency(engine.energy)
        self.energy_gradient.add_dependency(engine.energy_gradient)

    @staticmethod
    def _get_energy(step, energy, energy2=None, protocol=None, nrespa=None):
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
    def _get_energy_gradient(step, gradient, gradient2=None, protocol=None, nrespa=None):
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
