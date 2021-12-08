
import numpy as np


class Hydrogen:
    def __init__(self, units="atomic", azimuthal_number=0, energy=-1.0):
        self.units = units
        self.azimuthal_number = azimuthal_number  # quantum number l
        self.energy = energy  # ground state energy in atomic units

    def compute_radial_equation_error(self, r_data, coefficients, R_vector, T_vector):
        error = T_vector - self.g_radial_equation(r_data) * R_vector
        return error

    def g_radial_equation(self, r_data):
        num_data_points = np.shape(r_data)[0]
        energy_vector = np.full(num_data_points, self.energy)
        r_inverse_vector = np.array([1 / r for r in r_data])
        r_inverse_squared_vector = np.array([1 / r**2 for r in r_data])
        return (-2 * energy_vector + self.azimuthal_number * (self.azimuthal_number + 1) * r_inverse_squared_vector - 2 * r_inverse_vector)


def cost_function(coefficients, r_data, hydrogen):
    """Compute cost function.

    The radial function for Hydrogen is a second order ODE, which can be set to zero by bringing
    all the terms to one side. Our prediction for u(r) for each iteration can be substituted into
    this equation, and should be close to zero. This is how we define our error.

    Args:
        u_pred: (np.array)
        r_data: (np.array) a (1 x n) vector containing the input radial data
        R_matrix: (np.array) an (n x m) matrix, see ``create_R_matrix``
        T_matrix: (np.array) an (n x m) matrix, 2nd derivative, see ``create_T_matrix``
        coefficients: (np.array) a (1 x m) vector containing the predicted coefficients
        hydrogen: (Hydrogen) a Hydrogen object

    """
    T_vector = get_T_vector(r_data, coefficients)
    R_vector = get_R_vector(r_data, coefficients)
    error = hydrogen.compute_radial_equation_error(r_data, coefficients, R_vector, T_vector)
    squared_difference = np.dot(error, error)

    return squared_difference / len(r_data)


def cost_function_deriv(coefficients, r_data, hydrogen):
    T_vector = get_T_vector(r_data, coefficients)
    R_vector = get_R_vector(r_data, coefficients)
    error = hydrogen.compute_radial_equation_error(r_data, coefficients, R_vector, T_vector)
    num_data_points = np.shape(r_data)[0]
    J_deriv = np.zeros(3)
    J_deriv[0] = 2 / coefficients[0] * np.dot(error, error)
    J_deriv[1] = 2 / coefficients[2] * np.dot(error, error)
    g = hydrogen.g_radial_equation(r_data)
    for m in range(0, num_data_points):
        r = r_data[m]
        q = np.exp(-(r - coefficients[1]) / coefficients[2])
        u_deriv_theta_2 = r * q * (r - coefficients[1]) / (coefficients[2]**2)
        deriv_u_deriv_theta_2 = coefficients[0]**2 / coefficients[2]**4 * q * \
            (r * (coefficients[2] + coefficients[1] + 2 - coefficients[2] / coefficients[0]) - \
            r**2 - 2 * (coefficients[2] + 1))
        J_deriv[2] = J_deriv[2] + 2 * error[m] * (deriv_u_deriv_theta_2 - g[m] * u_deriv_theta_2)
    J_deriv = np.array(J_deriv)
    J_deriv = J_deriv / num_data_points
    return J_deriv


def ansatz_function(coefficients, r):
    return r * exponential_function(coefficients, r)


def exponential_function(coefficients, r):
    return coefficients[0] * np.exp(-(r - coefficients[1]) / coefficients[2])


def get_R_vector(r_data, coefficients):
    R_vector = []
    for r in r_data:
        R_vector.append(ansatz_function(coefficients, r))
    return np.array(R_vector)


def get_T_vector(r_data, coefficients):
    T_vector = []
    for r in r_data:
        T_vector.append((exponential_function(coefficients, r) / coefficients[2]) * (r / coefficients[2] - 2))
    return np.array(T_vector)


def gradient_descent(coefficients, alpha, max_cost, max_iterations, cost_function, cost_function_deriv, *args):

    for n in range(0, max_iterations):
        J_deriv = cost_function_deriv(coefficients, *args)
        cost = cost_function(coefficients, *args)
        # print(alpha * J_deriv)
        coefficients = coefficients - alpha * J_deriv
        if cost <= max_cost:
            break
    return coefficients, cost
