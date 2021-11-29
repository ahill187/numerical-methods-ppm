import numpy as np


class Hydrogen:
    def __init__(self, units="atomic", azimuthal_number=0, energy=1.0):
        self.units = units
        self.azimuthal_number = azimuthal_number  # quantum number l
        self.energy = energy  # ground state energy in atomic units

    def compute_radial_equation_error(self, r_data, R_matrix, T_matrix, coefficients):
        num_data_points = np.shape(R_matrix)[0]
        energy_vector = np.full(num_data_points, self.energy)
        r_inverse_vector = np.array([1/r for r in r_data])
        r_inverse_squared_vector = np.array([1/r**2 for r in r_data])
        error = np.matmul(T_matrix, coefficients) - np.matmul(
            (2*energy_vector + self.azimuthal_number*(self.azimuthal_number + 1)*r_inverse_squared_vector - r_inverse_vector),
            np.matmul(R_matrix, coefficients))
        return error


def cost_function(u_pred, r_data, R_matrix, T_matrix, coefficients, hydrogen):
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
    return squared_difference/len(r_data)


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
            T_matrix[row][n] = n*(n-1)*r**(n-2)
        row = row + 1
    return T_matrix


def main():

    polynomial_degree = 10
    r_data = np.arange(0.001, 20, 0.05)
    num_data_points = len(r_data)
    hydrogen = Hydrogen()
    R_matrix = create_R_matrix(num_data_points, polynomial_degree, r_data)
    T_matrix = create_T_matrix(num_data_points, polynomial_degree, r_data)
    coefficients = np.ones(polynomial_degree)
    u_pred = np.matmul(R_matrix, coefficients)
    cost = cost_function(u_pred, r_data, R_matrix, T_matrix, coefficients, hydrogen)


if __name__ == "__main__":
    main()
