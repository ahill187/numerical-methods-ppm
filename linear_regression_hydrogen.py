
import numpy as np


def cost_function_deriv(coefficients, r_data, hydrogen, u_pred, R_matrix, T_matrix):
    T_vector = get_T_vector(r_data, coefficients)
    R_vector = get_R_vector(r_data, coefficients)
    error = hydrogen.compute_radial_equation_error(r_data, coefficients, R_vector, T_vector)
    g_vector = hydrogen.g_radial_equation(r_data)
    polynomial_degree = np.shape(R_matrix)[1]
    num_data_points = np.shape(R_matrix)[0]
    J_deriv = []
    for k in range(0, polynomial_degree):
        identity = np.zeros(polynomial_degree)
        identity[k] = 1.0
        J_deriv_k = 2 / num_data_points * np.dot(error, np.matmul(T_matrix, identity) - \
                    np.matmul(T_matrix, identity) * g_vector)
        J_deriv.append(J_deriv_k)
    return np.array(J_deriv)


def create_R_matrix(num_data_points, polynomial_degree, r_data):
    """Create a matrix of polynomials.

        [
            1, r_1, r_1^2, ... r_1^m

            ...

            1, r_n, r_n^2, ... r_n^m

        ]

    Args:
        num_data_points: (int) number of data points to be used in the linear regression (n)
        polynomial_degree: (int) maximum degree of the polynomial (m)
        r_data: (np.array) a (1 x n) vector containing the input radial data

    """
    R_matrix = np.ones((num_data_points, polynomial_degree))
    row = 0
    for r in r_data:
        for n in range(0, polynomial_degree):
            R_matrix[row][n] = r**n
        row = row + 1
    return R_matrix


def create_T_matrix(num_data_points, polynomial_degree, r_data):
    """Create a matrix of second derivative of polynomials.

        [
            0, 0, 2, 6*r_1, 12*r_1^2... m*(m-1)*r_1^(m-2)

            ...

            0, 0, 2, 6*r_n, 12*r_n^2... m*(m-1)*r_n^(m-2)

        ]

    Args:
        num_data_points: (int) number of data points to be used in the linear regression (n)
        polynomial_degree: (int) maximum degree of the polynomial (m)
        r_data: (np.array) a (1 x n) vector containing the input radial data

    """
    T_matrix = np.ones((num_data_points, polynomial_degree))
    row = 0
    for r in r_data:
        for n in range(2, polynomial_degree):
            T_matrix[row][n] = n * (n - 1) * r**(n - 2)
        row = row + 1
    return T_matrix


def get_R_vector(r_data, coefficients):
    num_data_points = len(r_data)
    polynomial_degree = len(coefficients)
    R_matrix = create_R_matrix(num_data_points, polynomial_degree, r_data)
    R_vector = np.matmul(R_matrix, coefficients)
    return np.array(R_vector)


def get_T_vector(r_data, coefficients):
    num_data_points = len(r_data)
    polynomial_degree = len(coefficients)
    T_matrix = create_T_matrix(num_data_points, polynomial_degree, r_data)
    T_vector = np.matmul(T_matrix, coefficients)
    return np.array(T_vector)
