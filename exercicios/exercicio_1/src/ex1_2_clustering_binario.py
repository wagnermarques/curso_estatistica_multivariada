import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import os

def calcular_matriz_distancias(X, metric):
    """
    Calcula a matriz de distancias usando a metrica especificada.
    """
    return pdist(X, metric=metric)


def plot_dendrogram(dist_matrix, labels, metric, title, filename):
    """
    Realiza o agrupamento hierarquico a partir de uma matriz de distancias e plota o dendrograma.
    """
    # Agrupamento hierarquico (usando UPGMA - average linkage)
    Z = linkage(dist_matrix, method='average')
    
    # Geracao do grafico
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=12)
    plt.title(title, fontsize=14)
    plt.xlabel('Especies', fontsize=12)
    plt.ylabel(f'Distancia ({metric})', fontsize=12)
    plt.tight_layout()
    
    # Salvar no diretÃ³rio de figuras (text/figures)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'text', 'figures', filename)
    
    # Garante que o diretÃ³rio exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Dendrograma salvo em: {output_path}")

def analyze_binary_data(df):
    print("Iniciando analise de dados binÃ¡rios...")
    
    labels = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    print("\n--- Coeficiente Sorensen-Dice ---")
    dist_dice = calcular_matriz_distancias(X, 'dice')
    plot_dendrogram(dist_dice, labels, 'dice', 'Dendrograma - Sorensen-Dice', 'dendrograma_sorensen_dice.png')
    
    print("\n--- Coeficiente Simple Matching ---")
    dist_sm = calcular_matriz_distancias(X, 'matching')
    plot_dendrogram(dist_sm, labels, 'matching', 'Dendrograma - Simple Matching', 'dendrograma_simple_matching.png')
    
    print("\nAnÃ¡lise concluÃ­da.")

