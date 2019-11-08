from string import Template


template = """\
$$rem
${options}\
qm_mm true
igdefield 1
symmetry off
sym_ignore true
print_input false
qmmm_print true
skip_charge_self_interact true
$$end

"""


def get_qm_template(options_dict=None):

    options = {
        "jobtype": "force",
        "method": "hf",
        "basis": "6-31g*",
    }

    if options_dict is not None:
        options.update(options_dict)

    options = "".join([f"{key} {value}\n" for key, value in options.items()])

    return Template(template).safe_substitute(options=options)
