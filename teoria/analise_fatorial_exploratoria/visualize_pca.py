import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações de estilo
sns.set_theme(style="whitegrid")

# Localização dos arquivos
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
input_file = os.path.join(script_dir, 'dados_padronizados.csv')
img_dir = os.path.join(script_dir, 'imgs')

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

if os.path.exists(input_file):
    df_std = pd.read_csv(input_file)
    R = df_std.corr()
    eigenvalues, eigenvectors = np.linalg.eig(R)
    
    # Ordenar
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 1. Scree Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', linewidth=2)
    plt.title('Scree Plot - Variância por Componente')
    plt.xlabel('Componente Principal')
    plt.ylabel('Autovalor (Variância)')
    plt.axhline(y=1, color='r', linestyle='--', label='Critério de Kaiser (λ=1)')
    plt.legend()
    plt.savefig(os.path.join(img_dir, 'pca_scree_plot.png'))
    plt.close()

    # 2. Biplot (Cargas e Scores)
    # Cálculo dos scores: Z * V
    scores = df_std.values @ eigenvectors
    
    plt.figure(figsize=(10, 8))
    # Plotar os scores das amostras
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.5, color='gray', label='Amostras')
    
    # Plotar os vetores das variáveis (Cargas fatoriais: V * sqrt(L))
    loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    
    for i, var in enumerate(df_std.columns):
        plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, color='r', alpha=0.8, head_width=0.1)
        plt.text(loadings[i, 0]*3.5, loadings[i, 1]*3.5, var, color='darkred', ha='center', va='center', fontweight='bold')

    plt.title('PCA Biplot (PC1 vs PC2)')
    plt.xlabel(f'PC1 ({ (eigenvalues[0]/sum(eigenvalues))*100:.1f}%)')
    plt.ylabel(f'PC2 ({ (eigenvalues[1]/sum(eigenvalues))*100:.1f}%)')
    plt.grid(True)
    plt.savefig(os.path.join(img_dir, 'pca_biplot.png'))
    plt.close()
    
    print("Gráficos gerados com sucesso em teoria/analise_fatorial_exploratoria/imgs/")
