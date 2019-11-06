import numpy as np

try:
    from mdi import MDI_Init, MDI_Accept_Communicator, MDI_Recv_Command
    from mdi import MDI_Recv, MDI_Send
    from mdi import MDI_CHAR, MDI_DOUBLE, MDI_INT, MDI_DOUBLE_NUMPY
    use_mdi = True
except ImportError:
    use_mdi = False

from ..system import System


class IOMDI(object):

    def __init__(self, cwd=None):
        self.mode = "mdi"
        self.cwd = cwd

    def load_system(self, port, system=None, step=None):

        self._system = system

        if step is None:
            step = 0

        self._step = np.asarray(step)

        MDI_Init(f"-role ENGINE -name QMHub -method TCP -port {port} -hostname localhost", None)
        self.comm = MDI_Accept_Communicator()

        while True:
            command = MDI_Recv_Command(self.comm).strip()
            # print(f"Got a command from driver: {command}")

            if command == '<NAME':
                MDI_Send("QMHub", 1, MDI_CHAR, self.comm)

            elif command == '>NATOMS':
                self._n_atoms = MDI_Recv(1, MDI_INT, self.comm)

            elif command == '>NATOMS_QM':
                self._n_qm_atoms = MDI_Recv(1, MDI_INT, self.comm)

            elif command == '>QM_CHARGE':
                qm_charge = MDI_Recv(1, MDI_INT, self.comm)

            elif command == '>QM_MULT':
                qm_mult = MDI_Recv(1, MDI_INT, self.comm)

            elif command == '>CWD':
                self.cwd = MDI_Recv(1, MDI_CHAR, self.comm)

            elif command == '@INIT_MD':
                self._system = System(self._n_atoms, self._n_qm_atoms, qm_charge=qm_charge, qm_mult=qm_mult)
                return self._system

            elif command == "EXIT":
                break

            else:
                raise Exception(f"Unrecognized command: {command}")

    def return_results(self, energy, forces, outpus=None):
        assert self._system is not None

        while True:
            command = MDI_Recv_Command(self.comm).strip()
            # print(f"Got a command from driver: {command}")

            if command == "<ENERGY":
                MDI_Send(np.asscalar(energy), 1, MDI_DOUBLE, self.comm)

            elif command == "<FORCES":
                MDI_Send(forces.flatten(order="F"), 3 * self._n_atoms, MDI_DOUBLE_NUMPY, self.comm)

            elif command == ">STEP":
                self._step[()] = MDI_Recv(1, MDI_INT, self.comm)

            elif command == ">COORDS":
                self._system.atoms.positions[:] = MDI_Recv(3 * self._n_atoms, MDI_DOUBLE_NUMPY, self.comm).reshape(3, -1, order="F")

            elif command == ">CHARGES":
                self._system.atoms.charges[:] = MDI_Recv(self._n_atoms, MDI_DOUBLE_NUMPY, self.comm)

            elif command == ">QM_ELEMENTS":
                self._system.qm.atoms.elements[:] = MDI_Recv(self._n_qm_atoms, MDI_INT, self.comm)

            elif command == ">CELL":
                cell_basis = MDI_Recv(9, MDI_DOUBLE_NUMPY, self.comm).reshape(3, 3, order="F")
                cell_basis[np.isclose(cell_basis, 0.0)] = 0.0
                self._system.cell_basis[:] = cell_basis

            elif command == "EXIT":
                break

            else:
                raise Exception(f"Unrecognized command: {command}.")
