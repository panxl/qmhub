import numpy as np


def get_numerical_gradient(system, property, gradient, mask=None, over='i', i=None, j=None, k=None):
    ndim = property.ndim
    if mask is not None:
        j2 = np.where(mask)[0][j]
    else:
        j2 = j

    if ndim == 2:
        if over == 'i':
            analytical_gradient = -np.asscalar(gradient[k, i, j])

            system.atoms.positions[k, i:i+1] += 0.001
            property_pos = np.copy(property)

            system.atoms.positions[k, i:i+1] -= 0.002
            property_neg = np.copy(property)

            system.atoms.positions[k, i:i+1] += 0.001
        elif over == 'j':
            analytical_gradient = np.asscalar(gradient[k, i, j])

            system.atoms.positions[k, j2:j2+1] += 0.001
            property_pos = np.copy(property)

            system.atoms.positions[k, j2:j2+1] -= 0.002
            property_neg = np.copy(property)

            system.atoms.positions[k, j2:j2+1] += 0.001

        numerical_gradient = (property_pos[i, j] - property_neg[i, j]) / 0.002

    elif ndim == 1:
        if len(property) == 1:
            if over == 'i':
                analytical_gradient = np.asscalar(gradient[k, i])

                system.atoms.positions[k, i:i+1] += 0.001
                property_pos = np.copy(property)

                system.atoms.positions[k, i:i+1] -= 0.002
                property_neg = np.copy(property)

                system.atoms.positions[k, i:i+1] += 0.001

                numerical_gradient = (property_pos - property_neg) / 0.002
            elif over == 'j':
                analytical_gradient = np.asscalar(gradient[k, j])

                system.atoms.positions[k, j2:j2+1] += 0.001
                property_pos = np.copy(property)

                system.atoms.positions[k, j2:j2+1] -= 0.002
                property_neg = np.copy(property)

                system.atoms.positions[k, j2:j2+1] += 0.001

                numerical_gradient = (property_pos - property_neg) / 0.002
        elif len(property) == len(system.qm.atoms):
            if over == 'i':
                analytical_gradient = np.asscalar(gradient[k, i])

                system.atoms.positions[k, i:i+1] += 0.001
                property_pos = np.copy(property)

                system.atoms.positions[k, i:i+1] -= 0.002
                property_neg = np.copy(property)

                system.atoms.positions[k, i:i+1] += 0.001

                numerical_gradient = (property_pos[i] - property_neg[i]) / 0.002
            elif over == 'j':
                return
        else:
            if over == 'i':
                analytical_gradient = -np.asscalar(gradient[k, i, j])

                system.atoms.positions[k, i:i+1] += 0.001
                property_pos = np.copy(property)

                system.atoms.positions[k, i:i+1] -= 0.002
                property_neg = np.copy(property)

                system.atoms.positions[k, i:i+1] += 0.001

                numerical_gradient = (property_pos[j] - property_neg[j]) / 0.002
            elif over == 'j':
                analytical_gradient = np.asscalar(gradient[k, :, j].sum())

                system.atoms.positions[k, j2:j2+1] += 0.001
                property_pos = np.copy(property)

                system.atoms.positions[k, j2:j2+1] -= 0.002
                property_neg = np.copy(property)

                system.atoms.positions[k, j2:j2+1] += 0.001

                numerical_gradient = (property_pos[j] - property_neg[j]) / 0.002

    return analytical_gradient, numerical_gradient, analytical_gradient - numerical_gradient
