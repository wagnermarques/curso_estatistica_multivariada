import pandas as pd
import numpy as np
import os

# Localização do script e arquivos
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
input_file = os.path.join(script_dir, 'dados_padronizados.csv')

if not os.path.exists(input_file):
    print(f"Erro: Arquivo {input_file} não encontrado. Execute o script de padronização primeiro.")
else:
    # Carregar dados padronizados
    df_std = pd.read_csv(input_file)
    
    # Calcular matriz de correlação (R)
    R = df_std.corr()
    
    # Extrair autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eig(R)
    
    # Ordenar de forma decrescente
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("=== Resultados da Decomposição Espectral ===")
    print("\nAutovalores (Variância Explicada por cada componente):")
    for i, val in enumerate(eigenvalues):
        variancia_perc = (val / len(eigenvalues)) * 100
        print(f"Componente {i+1}: {val:.4f} ({variancia_perc:.2f}%)")
        
    print("\nMatriz de Autovetores (V) - Primeiros 2 componentes:")
    df_v = pd.DataFrame(eigenvectors[:, :2], index=df_std.columns, columns=['V1', 'V2'])
    print(df_v)
