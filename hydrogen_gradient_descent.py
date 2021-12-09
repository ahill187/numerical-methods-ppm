import numpy as np


class Hydrogen:
    def __init__(self, units="atomic", principal_number=1, azimuthal_number=0):
        self.units = units
        self.principal_number = principal_number  # quantum number n
        self.azimuthal_number = azimuthal_number  # quantum number l
        self.ground_state_energy = -1.0  # ground state energy in atomic units
        self.energy = self.ground_state_energy / (self.principal_number**2)  # energy in atomic units

    def compute_radial_equation_error(self, r_data, coefficients, R_vector, T_vector):
        error = T_vector - self.g_radial_equation(r_data) * R_vector
        return error

    def g_radial_equation(self, r_data):
        num_data_points = np.shape(r_data)[0]
        energy_vector = np.full(num_data_points, self.energy)
        r_inverse_vector = np.array([1 / r for r in r_data])
        r_inverse_squared_vector = np.array([1 / r**2 for r in r_data])
        return (-2 * energy_vector + self.azimuthal_number * (self.azimuthal_number + 1) * r_inverse_squared_vector - 2 * r_inverse_vector)


def cost_function(coefficients, r_data, hydrogen, get_R_vector, get_T_vector):
    """Compute cost function.

    The radial function for Hydrogen is a second order ODE, which can be set to zero by bringing
    all the terms to one side. Our prediction for u(r) for each iteration can be substituted into
    this equation, and should be close to zero. This is how we define our error.

    Args:
        coefficients: (np.array) a (1 x m) vector containing the predicted coefficients
        r_data: (np.array) a (1 x n) vector containing the input radial data
        hydrogen: (Hydrogen) a Hydrogen object
        get_R_vector: (callable) a function that takes (r_data, coefficients) as arguments and
            returns the vector containing u(r) = rR(r), where R(r) is the radial wavefunction.
        get_T_vector: (callable) a function that takes (r_data, coefficients) as arguments and
            returns the vector containing u"(r), where u(r) = rR(r) and R(r) is the radial
            wavefunction.

    """
    T_vector = get_T_vector(r_data, coefficients)
    R_vector = get_R_vector(r_data, coefficients)
    error = hydrogen.compute_radial_equation_error(r_data, coefficients, R_vector, T_vector)
    squared_difference = np.dot(error, error)
    return squared_difference / len(r_data)


def gradient_descent(coefficients, alpha, max_cost, max_iterations, r_data,
                     hydrogen, get_R_vector, get_T_vector, cost_function_deriv, *args):
    for n in range(0, max_iterations):
        J_deriv = cost_function_deriv(coefficients, r_data, hydrogen, *args)
        cost = cost_function(coefficients, r_data, hydrogen, get_R_vector, get_T_vector)
        coefficients = coefficients - alpha * J_deriv
        if cost <= max_cost:
            break
    return coefficients, cost
