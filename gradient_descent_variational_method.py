
import numpy as np


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
