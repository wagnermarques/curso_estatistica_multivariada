import sympy as sp
import sys

def get_expected_formula():
    E_ij, R_i, C_j, n = sp.symbols('E_{ij} R_i C_j n')
    return sp.Eq(E_ij, (R_i * C_j) / n)

def get_contribution_formula():
    O_ij, E_ij = sp.symbols('O_{ij} E_{ij}')
    return (O_ij - E_ij)**2 / E_ij

def get_chi2_stat_formula():
    chi2 = sp.symbols(r'\chi^2')
    O_ij, E_ij = sp.symbols('O_{ij} E_{ij}')
    i, j, r, c = sp.symbols('i j r c')
    # Summation formula
    summation = sp.Sum((O_ij - E_ij)**2 / E_ij, (i, 1, r), (j, 1, c))
    return sp.Eq(chi2, summation)

def main():
    formulas = {
        "qui2-expected-values": get_expected_formula,
        "qui2-contribution": get_contribution_formula,
        "qui2-stat": get_chi2_stat_formula
    }

    raw_mode = "--raw" in sys.argv
    if raw_mode:
        sys.argv.remove("--raw")

    if len(sys.argv) < 2 or sys.argv[1] not in formulas:
        print("Usage: ./venv/bin/python3 scripts/math_latex.py [qui2-expected-values|qui2-contribution|qui2-stat] [--raw]")
        print(f"Available formulas: {', '.join(formulas.keys())}")
        sys.exit(1)

    formula_name = sys.argv[1]
    formula_obj = formulas[formula_name]()
    
    latex_str = sp.latex(formula_obj)

    if raw_mode:
        print(latex_str)
    else:
        print(f"--- {formula_name.replace('_', ' ').title()} Formula (LaTeX) ---")
        print(latex_str)

if __name__ == "__main__":
    main()
