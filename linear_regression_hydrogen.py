
import numpy as np
import math


class Hydrogen:
    def __init__(self, units="atomic", principal_number=1,azimuthal_number=0, energy=-1.0):
        self.units = units
        self.azimuthal_number = azimuthal_number  # quantum number l
        self.principal_number = principal_number
        self.energy = energy/(self.principal_number**2)  # energy in atomic units

    def compute_radial_equation_error(self, r_data, R_matrix, T_matrix, coefficients):
        error = np.matmul(T_matrix, coefficients) - self.g_radial_equation(r_data) * np.matmul(R_matrix, coefficients)
        return error

    def g_radial_equation(self, r_data):
        num_data_points = np.shape(r_data)[0]
        energy_vector = np.full(num_data_points, self.energy)
        r_inverse_vector = np.array([1 / r for r in r_data])
        r_inverse_squared_vector = np.array([1 / r**2 for r in r_data])
        return (-2 * energy_vector + self.azimuthal_number * (self.azimuthal_number + 1) * r_inverse_squared_vector - 2 * r_inverse_vector)


def cost_function(coefficients, u_pred, r_data, R_matrix, T_matrix, hydrogen):
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
    error = hydrogen.compute_radial_equation_error(r_data, R_matrix, T_matrix, coefficients)
    squared_difference = np.dot(error, error)
    return squared_difference / len(r_data)


def cost_function_deriv(coefficients, u_pred, r_data, R_matrix, T_matrix, hydrogen):
    error = hydrogen.compute_radial_equation_error(r_data, R_matrix, T_matrix, coefficients)
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


def gradient_descent(coefficients, alpha, max_cost, max_iterations, cost_function, cost_function_deriv, *args):

    for n in range(0, max_iterations):
        J_deriv = cost_function_deriv(coefficients, *args)
        cost = cost_function(coefficients, *args)
        # print(alpha * J_deriv)
        coefficients = coefficients - alpha * J_deriv
        if cost <= max_cost:
            break
    return coefficients, cost


# def main():

#     polynomial_degree = 10
#     r_data = np.arange(0.001, 20, 0.05)
#     num_data_points = len(r_data)
#     hydrogen = Hydrogen()
#     R_matrix = create_R_matrix(num_data_points, polynomial_degree, r_data)
#     T_matrix = create_T_matrix(num_data_points, polynomial_degree, r_data)
#     coefficients = np.ones(polynomial_degree)
#     for n in range(0, polynomial_degree):
#         coefficients[n] = (-1)**n * (1 / math.factorial(n)) * 1 / np.sqrt(np.pi)
#     u_pred = np.matmul(R_matrix, coefficients)
#     coefficients, cost = gradient_descent(coefficients, alpha, max_cost, max_iterations, cost_function,
#                                     cost_function_deriv, u_pred, r_data, R_matrix, T_matrix,
#                                     ,hydrogen)




#
# if __name__ == "__main__":
#     main()
