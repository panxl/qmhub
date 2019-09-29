import importlib


QMCLASS = {'qchem': "QChem", 'orca': "ORCA", 'sqm': "SQM"}
QMMODULE = {'QChem': ".qchem", 'ORCA': ".orca", 'SQM': ".sqm"}


def choose_qmtool(qmSoftware):
    try:
        qm_class = QMCLASS[qmSoftware.lower()]
        qm_module = QMMODULE[qm_class]
    except:
        raise ValueError("Please choose 'qchem', 'orca', or 'sqm' for qmSoftware.")

    qmtool = importlib.import_module(qm_module, package='qmhub.qmtools').__getattribute__(qm_class)

    return qmtool
