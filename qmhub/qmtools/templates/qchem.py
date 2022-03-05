from string import Template


template = """\
$$rem
${options}\
qm_mm true
!igdefield 1
symmetry off
sym_ignore true
print_input false
qmmm_print true
skip_charge_self_interact true
$$end

"""


default_options = {
    "jobtype": "force",
    "method": "hf",
    "basis": "6-31g*",
}


def get_qm_template(options=None):

    options = options or default_options
    options = "".join([f"{key} {value}\n" for key, value in options.items()])

    return Template(template).safe_substitute(options=options)
