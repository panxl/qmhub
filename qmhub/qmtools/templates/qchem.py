from string import Template


template = """\
$$rem
${keywords}\
qm_mm true
igdefield 1
symmetry off
sym_ignore true
print_input false
qmmm_print true
skip_charge_self_interact true
$$end

"""


def get_qm_template(keywords_dict=None):

    keywords = {
        "jobtype": "force",
        "method": "b3lyp",
        "basis": "6-31g*",
    }

    if keywords_dict is not None:
        keywords.update(keywords_dict)

    keywords = "".join([f"{key} {value}\n" for key, value in keywords.items()])

    return Template(template).safe_substitute(keywords=keywords)
