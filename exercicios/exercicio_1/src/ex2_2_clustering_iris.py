import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import os

def plot_dendrogram_iris(dist_matrix, labels, title, filename):
    """
    Produces a customized dendrogram for the IRIS dataset using Ward's method.
    """
    # Agrupamento hierárquico (Ward is standard for numerical scaled data)
    Z = linkage(dist_matrix, method='ward')
    
    plt.figure(figsize=(15, 7))
    
    # Custom color map for species if labels are provided
    # (Simplified: using default dendrogram coloring)
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Observações / Espécies', fontsize=12)
    plt.ylabel('Distância de Ward', fontsize=12)
    plt.tight_layout()
    
    # Salvar no diretório de figuras (text/figures)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'text', 'figures', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Dendrograma IRIS salvo em: {output_path}")
