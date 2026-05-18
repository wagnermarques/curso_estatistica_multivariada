import os
import sys
import pandas as pd
import numpy as np

# Adicionando o caminho das libs para reuso
LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../libs'))

if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)

from data_io.data_loader import load_pca_data

def main():
    # Configuracoes de exibicao do Pandas
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    print("=== Passo 1: Carregamento de Dados ===")
    csv_path = os.path.join(os.path.dirname(__file__), 'exemplo_pca.csv')
    try:
        data = load_pca_data(csv_path)
        print("Dados originais (primeiras linhas):")
        print(data.head())
        
        X = data.values
        
        print("\n=== Passo 2: Padronizacao (Z-score) ===")
        # Padronizacao: (X - media) / desvio_padrao
        # ddof=1 para desvio padrao amostral (corrigido)
        mean_X = np.mean(X, axis=0)

        print("\n=== Passo 2.1: Media de cada coluna ===")
        for i, col in enumerate(data.columns):
            print(f"{col}: {mean_X[i]:.4f}")
            
        std_X = np.std(X, axis=0, ddof=1)
        X_std = (X - mean_X) / std_X
        
        print("Dados Padronizados (X_std):")
        print(pd.DataFrame(X_std, columns=data.columns).head())

        print("\n=== Passo 3: Matriz de Covariancia, Autovalores e Autovetores ===")
        # A matriz de covariancia de dados padronizados e a matriz de correlacao
        cov_mat = np.cov(X_std.T)
        print("Matriz de Covariancia (R):")
        print(cov_mat)

        # Calculo de Autovalores e Autovetores
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        print("\nAutovalores (Variancia):")
        print(eig_vals)

        print("\nAutovetores (Cargas dos Componentes):")
        print(eig_vecs)

        # Variancia Explicada
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
        
        print("\nVariancia Explicada Acumulada:")
        cum_var_exp = np.cumsum(var_exp)
        for i, (v, cv) in enumerate(zip(var_exp, cum_var_exp)):
            print(f"CP{i+1}: Individual = {v:.2f}% | Acumulada = {cv:.2f}%")

    except Exception as e:
        print(f"Erro durante a execucao: {e}")

if __name__ == "__main__":
    main()
