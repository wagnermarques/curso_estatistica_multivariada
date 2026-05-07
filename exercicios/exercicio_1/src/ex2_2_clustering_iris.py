import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

def plot_elbow_method(data, max_k=10, filename='kmeans_elbow_iris.png'):
    """
    Plots the elbow method to help find the optimal number of clusters.
    """
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--', color='b')
    plt.title('Metodo do Cotovelo (Elbow Method) - IRIS', fontsize=14, fontweight='bold')
    plt.xlabel('Numero de Grupos (k)', fontsize=12)
    plt.ylabel('WCSS (Soma dos Quadrados Intra-Cluster)', fontsize=12)
    plt.grid(True)
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'text', 'figures', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafico do Cotovelo salvo em: {output_path}")

def executar_kmeans(data, labels, n_clusters=3):
    """
    Executa o algoritmo K-means e retorna um DataFrame com as atribuicoes.
    """
    #Preciso entender melhor ainda max_iter
    # max_iter = numero maximo de iteracoes para o algoritmo convergir, ou seja, para encontrar os centroides finais. 
    # Se o algoritmo atingir esse numero de iteracoes sem convergir, ele para e retorna os resultados atuais. 
    # O valor padrao é 300, mas pode ser ajustado dependendo do tamanho do dataset e da complexidade dos dados.
    # a iteração significa o processo de atribuir cada ponto de dados ao cluster mais próximo e, em seguida, 
    # recalcular os centroides com base nos pontos atribuídos a cada cluster.

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    # k-means++ é uma técnica de inicialização que melhora a escolha dos 
    # centroides iniciais, o que pode levar a uma convergência 
    # mais rápida e a resultados mais estáveis.
    # outras opções são: 
    # 'random' (centroides iniciais escolhidos aleatoriamente) ou uma matriz de centroides pré-definida.
    # ja o kmeans++, operacionalmente falando, funciona da seguinte maneira:
    # 1. O primeiro centroide é escolhido aleatoriamente a partir dos dados.
    # 2. Para cada ponto de dados, calcula-se a distância ao centroide mais próximo.
    # 3. O próximo centroide é escolhido com uma probabilidade proporcional ao quadrado da distância ao centroide mais próximo, o que significa que pontos mais distantes têm uma chance maior de serem escolhidos como centroides iniciais.
    # 4. Esse processo é repetido até que todos os centroides iniciais sejam escolhidos.

    cluster_ids = kmeans.fit_predict(data)

    df_grupos = pd.DataFrame({
        'Especie': labels,
        'Grupo': cluster_ids + 1  # Ajustando para comecar em 1
    })
    # retornamos
    # df_grupos: um DataFrame contendo as espécies originais e os grupos atribuídos pelo K-means.
    # kmeans: o modelo KMeans treinado, que inclui os centroides e outras informações sobre o ajuste do modelo.
    return df_grupos, kmeans

def plot_kmeans_clusters(data, cluster_labels, centroids, filename='kmeans_scatter_iris.png'):
    """
    Visualiza os grupos do K-means usando as duas primeiras variaveis padronizadas.
    """
    plt.figure(figsize=(10, 7))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i in range(1, len(np.unique(cluster_labels)) + 1):
        plt.scatter(
            data[cluster_labels == i, 0], 
            data[cluster_labels == i, 1], 
            s=70, 
            c=colors[i-1 % len(colors)], 
            label=f'Grupo {i}',
            alpha=0.6,
            edgecolors='w'
        )

    # Plotting the centroids
    plt.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        s=250, 
        c='yellow', 
        label='Centroides', 
        marker='*', 
        edgecolors='black'
    )
    
    plt.title('Clusters K-means (Projecao: Variavel 1 vs Variavel 2)', fontsize=14, fontweight='bold')
    plt.xlabel('Variavel 1 (Padronizada)', fontsize=12)
    plt.ylabel('Variavel 2 (Padronizada)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'text', 'figures', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Grafico de Clusters (Scatter) salvo em: {output_path}")
