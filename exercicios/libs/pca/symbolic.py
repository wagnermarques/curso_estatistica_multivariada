from sympy import symbols, Matrix, simplify, Rational, sqrt, Symbol

def get_symbolic_standardization():
    """
    Returns symbolic representations for standardization.
    """
    x, mu, sigma = symbols('x mu sigma')
    z = (x - mu) / sigma
    return x, mu, sigma, z

def get_symbolic_covariance_matrix(n_vars=3):
    """
    Returns a symbolic covariance matrix.
    """
    # Create a symmetric matrix of symbols c_ij
    C = Matrix(n_vars, n_vars, lambda i, j: Symbol(f'c_{min(i,j)+1}{max(i,j)+1}'))
    return C

def get_standardization_formula():
    """
    Returns the LaTeX string for the standardization formula.
    """
    return r"Z = \frac{X - \mu}{\sigma}"

def get_covariance_formula():
    """
    Returns the LaTeX string for the covariance matrix formula.
    """
    return r"C = \frac{1}{n-1} Z^T Z"
