import sys
import os
from sympy import pprint, init_printing

# Add the libs directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exercicios/libs')))

from pca.symbolic import get_symbolic_standardization, get_symbolic_covariance_matrix

def main():
    print("Step 2: Symbolic Math with SymPy\n")
    
    # Initialize pretty printing
    init_printing(use_unicode=True)
    
    print("1. Standardization Formula:")
    x, mu, sigma, z = get_symbolic_standardization()
    pprint(z)
    print("\n")
    
    print("2. Symbolic Covariance Matrix (3x3):")
    C = get_symbolic_covariance_matrix(3)
    pprint(C)
    print("\n")

if __name__ == "__main__":
    main()
