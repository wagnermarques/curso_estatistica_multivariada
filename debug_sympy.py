import sympy as sp
import os

try:
    a, b, c = sp.symbols('a b c')
    S_sd = (2*a) / (2*a + b + c)
    expr_latex = r"$S_{SD} = " + sp.latex(S_sd) + r"$"
    
    # Tentativa manual de preview para ver erro
    sp.preview(expr_latex, viewer='file', filename='test_formula.png', euler=False)
    print(f"File size: {os.path.getsize('test_formula.png')} bytes")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    if os.path.exists('test_formula.png') and os.path.getsize('test_formula.png') == 0:
        print("Test file is 0 bytes.")
