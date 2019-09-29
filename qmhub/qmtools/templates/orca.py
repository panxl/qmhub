from string import Template


template = """\
! ${keywords}KeepDens
%output PrintLevel Mini Print[ P_Mulliken ] 1 Print[P_AtCharges_M] 1 end
%pal nprocs ${nproc} end
%pointcharges "${pointcharges}"
"""


def get_qm_template(keywords_dict=None, nproc=None, pointcharges=None):

    keywords = {
        "jobtype": "EnGrad",
        "method": "HF",
        "basis": "6-31G(d)",
        "grid": "Grid4 NOFINALGRID",
        "scf_convergence": "TightSCF",
        "scf_guess": "NoAutoStart",
    }

    if keywords_dict is not None:
        keywords.update(keywords_dict)

    keywords = "".join([f"{value} " for value in keywords.values()])

    if nproc is None:
        nproc = 1
    
    if pointcharges is None:
        pointcharges = "orca.pc"

    return Template(template).safe_substitute(keywords=keywords, nproc=nproc, pointcharges=pointcharges)
