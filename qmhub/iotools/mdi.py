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
        self.node = "@GLOBAL"

        while True:
            command = MDI_Recv_Command(self.comm)
            # print("Got a command from driver: %s" % command)

            if command.strip() == "<@":
                MDI_Send(self.node, 1, MDI_CHAR, self.comm)

            elif command.strip() == '<NAME':
                MDI_Send("QMHub", 1, MDI_CHAR, self.comm)

            elif command.strip() == '>NATOMS_QM':
                n_qm_atoms = MDI_Recv(1, MDI_INT, self.comm)

            elif command.strip() == '>NATOMS_MM':
                n_mm_atoms = MDI_Recv(1, MDI_INT, self.comm)

            elif command.strip() == '>QM_CHARGE':
                qm_charge = MDI_Recv(1, MDI_INT, self.comm)

            elif command.strip() == '>QM_MULT':
                qm_mult = MDI_Recv(1, MDI_INT, self.comm)

            elif command.strip() == '>CWD':
                self.cwd = MDI_Recv(1, MDI_CHAR, self.comm)

            elif command.strip() == '@INIT_MD':
                n_atoms = n_qm_atoms + n_mm_atoms
                self._system = System(n_atoms, n_qm_atoms, qm_charge=qm_charge, qm_mult=qm_mult)
                self.node = "@INIT_MD"
                return self._system

            else:
                raise Exception(f"Unrecognized command: {command.strip()}")

    def return_results(self, energy, forces, outpus=None):
        assert self.node == "@INIT_MD"

        while True:
            command = MDI_Recv_Command(self.comm)
            # print("Got a command from driver: %s" % command)

            if command.strip() == "<@":
                MDI_Send(self.node, 1, MDI_CHAR, self.comm)

            elif command.strip() == "@FORCES":
                energy.update_cache()
                forces.update_cache()
                self.node = "@FORCES"

            elif command.strip() == "@COORDS":
                self.node = "@COORDS"

            elif command.strip() == "<ENERGY":
                MDI_Send(np.asarray(energy), 1, MDI_DOUBLE, self.comm)

            elif command.strip() == "<QM_FORCES":
                n_qm_atoms = len(self._system.qm.atoms)
                MDI_Send(forces[:, :n_qm_atoms].flatten(order="F"), 3 * n_qm_atoms, MDI_DOUBLE_NUMPY, self.comm)

            elif command.strip() == "<MM_FORCES":
                n_mm_atoms = len(self._system.mm.atoms)
                MDI_Send(forces[:, -n_mm_atoms:].flatten(order="F"), 3 * n_mm_atoms, MDI_DOUBLE_NUMPY, self.comm)

            elif command.strip() == ">QM_COORDS":
                n_qm_atoms = len(self._system.qm.atoms)
                self._system.qm.atoms.positions[:] = MDI_Recv(3 * n_qm_atoms, MDI_DOUBLE_NUMPY, self.comm).reshape(3, -1, order="F")

            elif command.strip() == ">QM_CHARGES":
                n_qm_atoms = len(self._system.qm.atoms)
                self._system.qm.atoms.charges[:] = MDI_Recv(n_qm_atoms, MDI_DOUBLE_NUMPY, self.comm)

            elif command.strip() == ">QM_ELEMENTS":
                n_qm_atoms = len(self._system.qm.atoms)
                self._system.qm.atoms.elements[:] = MDI_Recv(n_qm_atoms, MDI_INT, self.comm)

            elif command.strip() == ">MM_COORDS":
                n_mm_atoms = len(self._system.mm.atoms)
                self._system.mm.atoms.positions[:] = MDI_Recv(3 * n_mm_atoms, MDI_DOUBLE_NUMPY, self.comm).reshape(3, -1, order="F")

            elif command.strip() == ">MM_CHARGES":
                n_mm_atoms = len(self._system.mm.atoms)
                self._system.mm.atoms.charges[:] = MDI_Recv(n_mm_atoms, MDI_DOUBLE_NUMPY, self.comm)

            elif command.strip() == ">CELL":
                cell_basis = MDI_Recv(9, MDI_DOUBLE_NUMPY, self.comm).reshape(3, 3, order="F")
                cell_basis[np.isclose(cell_basis, 0.0)] = 0.0
                self._system.cell_basis[:] = cell_basis

            elif command.strip() == ">STEP":
                self._step[()] = MDI_Recv(1, MDI_INT, self.comm)

            elif command.strip() == "EXIT":
                break

            else:
                raise Exception(f"Unrecognized command: {command.strip()}.")
