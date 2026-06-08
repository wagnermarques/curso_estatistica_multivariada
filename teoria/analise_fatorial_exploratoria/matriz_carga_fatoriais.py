import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
input_file = os.path.join(script_dir, 'dados_padronizados.csv')

if os.path.exists(input_file):
    df_std = pd.read_csv(input_file)
    R = df_std.corr()
    eigenvalues, eigenvectors = np.linalg.eig(R)
    
    # Ordenar
    idx = eigenvalues.argsort()[::-1]
    L = eigenvalues[idx]
    V = eigenvectors[:, idx]
    
    # Pegar apenas os 2 primeiros (Kaiser > 1)
    L_2 = L[:2]
    V_2 = V[:, :2]
    
    # Calcular A = V * sqrt(L)
    # np.diag(np.sqrt(L_2)) cria a matriz diagonal com as raízes
    A = V_2 @ np.diag(np.sqrt(L_2))
    
    # Criar DataFrame para visualização
    df_a = pd.DataFrame(A, index=df_std.columns, columns=['Fator 1', 'Fator 2'])
    
    # Calcular Comunalidade (Soma dos quadrados das cargas na linha)
    df_a['Comunalidade (h2)'] = (df_a**2).sum(axis=1)
    
    print("=== Matriz de Cargas Fatoriais (Calculada) ===")
    print(df_a.round(3))
