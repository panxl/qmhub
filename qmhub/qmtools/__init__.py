import importlib

QMCLASS = {'qchem': "QChem", 'dftbplus': "DFTB", 'orca': "ORCA",
           'mopac': "MOPAC", 'psi4': "PSI4", 'sqm': "SQM"}

QMMODULE = {'QChem': ".qchem", 'DFTB': ".dftb", 'ORCA': ".orca",
            'MOPAC': ".mopac", 'PSI4': ".psi4", 'SQM': ".sqm"}


def choose_qmtool(qmSoftware):
    try:
        qm_class = QMCLASS[qmSoftware.lower()]
        qm_module = QMMODULE[qm_class]
    except:
        raise ValueError("Please choose 'qchem', 'dftbplus', 'orca', 'mopac', 'psi4', or 'sqm' for qmSoftware.")

    qmtool = importlib.import_module(qm_module, package='qmhub.qmtools').__getattribute__(qm_class)

    return qmtool