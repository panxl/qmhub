import importlib


QM_TO_CLASS_MAP = {
    'qchem': "QChem",
    'orca': "ORCA",
    'sqm': "SQM",
}


class QM(object):
    @classmethod
    def create(cls, qm_name, *args, **kwargs):
        if qm_name not in QM_TO_CLASS_MAP:
            raise ValueError("Please choose 'qchem', 'orca', or 'sqm' for QM engine.")

        qm_module = importlib.import_module("qmhub.qmtools." + qm_name)
        qm_cls = qm_module.__getattribute__(QM_TO_CLASS_MAP[qm_name])

        return qm_cls(*args, **kwargs)
