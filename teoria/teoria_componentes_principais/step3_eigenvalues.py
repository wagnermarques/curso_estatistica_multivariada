import sys
import os
import numpy as np

# Add the libs directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exercicios/libs')))

from data_io.data_loader import load_pca_data

def main():
    print("Passo 3: Autovalores e Autovetores\n")
    
    csv_path = os.path.join(os.path.dirname(__file__), 'exemplo_pca.csv')
    
    try:
        # Carregar dados
        data = load_pca_data(csv_path)
        X = data.values

        # Padronizar os dados
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

        # Matriz de Covariancia
        cov_mat = np.cov(X_std.T)

        # Autovalores e Autovetores
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        print("Autovalores:")
        print(eig_vals)

        print("\nAutovetores (Componentes):")
        print(eig_vecs)

        # Variancia Explicada
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
        
        print("\nVariancia Explicada por cada Componente (%):")
        for i, v in enumerate(var_exp):
            print(f"CP{i+1}: {v:.2f}%")
            
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
