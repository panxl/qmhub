from string import Template


template = """\
! ${options}KeepDens
%output PrintLevel Mini Print[ P_Mulliken ] 1 Print[P_AtCharges_M] 1 end
%pal nprocs ${nproc} end
%pointcharges "${pointcharges}"
"""


default_options = {
    "jobtype": "EnGrad",
    "method": "HF",
    "basis": "6-31G(d)",
    "grid": "Grid4 NOFINALGRID",
    "scf_convergence": "TightSCF",
    "scf_guess": "NoAutoStart",
}


def get_qm_template(options=None, nproc=None, pointcharges=None):

    if options is None:
        options = default_options

    options = "".join([f"{value} " for value in options.values()])

    if nproc is None:
        nproc = 1
    
    if pointcharges is None:
        pointcharges = "orca.pc"

    return Template(template).safe_substitute(options=options, nproc=nproc, pointcharges=pointcharges)
