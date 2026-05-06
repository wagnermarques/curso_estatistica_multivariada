import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import os

def plot_dendrogram_iris(dist_matrix, labels, title, filename):
    """
    Produces a customized dendrogram for the IRIS dataset using Ward's method.
    """
    # Agrupamento hierÃ¡rquico (Ward is standard for numerical scaled data)
    Z = linkage(dist_matrix, method='ward')

    plt.figure(figsize=(15, 7))

    # Custom color map for species if labels are provided
    # (Simplified: using default dendrogram coloring)
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Observacoes / Especies', fontsize=12)
    plt.ylabel('Distancia de Ward', fontsize=12)
    plt.tight_layout()

    # Salvar no diretÃ³rio de figuras (text/figures)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'text', 'figures', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Dendrograma IRIS salvo em: {output_path}")

def atribuir_grupos(dist_matrix, labels, t, criterion='distance', method='ward'):
    """
    Corta o dendrograma e atribui grupos as amostras.
    criterion pode ser 'distance' (corte na altura t) ou 'maxclust' (t grupos).
    """
    Z = linkage(dist_matrix, method=method)
    cluster_ids = fcluster(Z, t=t, criterion=criterion)

    df_grupos = pd.DataFrame({
        'Especie': labels,
        'Grupo': cluster_ids
    })
    return df_grupos

