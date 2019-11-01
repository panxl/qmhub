import numpy as np


class Simulation(object):
    def __init__(self, protocol=None, engine_name=None, engine2_name=None, *, nrespa=None):
        if protocol is not None:
            self.protocol = protocol
        else:
            self.protocol = "md"

        if engine_name is not None:
            self.engine_name = engine_name
        else:
            self.engine_name = "engine"

        self.engine2_name = engine2_name

        if self.protocol.lower() == "mts":
            if self.engine2_name is None:
                self.engine2_name = "engine2"

            if nrespa is not None:
                self.nrespa = nrespa
            else:
                raise ValueError("Please set 'nrespa' for 'mts' protocol.")

        self.step = 0

    def add_engine(self, name, engine):
        if name in [self.engine_name, self.engine2_name]:
            setattr(self, name, engine)

    def return_results(self):
        if not hasattr(self, self.engine_name):
            raise AttributeError("Please add engine first.")

        if self.protocol.lower() == "md":
            engine = getattr(self, self.engine_name)
            return engine.energy, engine.energy_gradient
        elif self.protocol.lower() == "mts":
            if not hasattr(self, self.engine2_name):
                raise AttributeError("Please add engine2 first.")

            if self.step % self.nrespa == 0:
                engine = getattr(self, self.engine_name)
                engine2 = getattr(self, self.engine2_name)
                return (
                    engine.energy,
                    engine2.energy_gradient + self.nrespa * (
                        engine.energy_gradient
                        - engine.energy_gradient
                    )
                )
            else:
                engine2 = getattr(self, self.engine2_name)
                return 0., engine2.energy_gradient
        else:
            raise ValueError("Only 'md' and 'mts' are supported.")
