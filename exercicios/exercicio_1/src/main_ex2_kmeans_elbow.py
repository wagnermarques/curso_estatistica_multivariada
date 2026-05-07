import ex2_1_load_iris
import ex2_2_clustering_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Leitura dos dados
df = ex2_1_load_iris.load_iris_data()

# 2. Mostrar as primeiras 10 linhas
print("--- Primeiras 10 linhas do dataset IRIS ---")
print(df.head(10))

# 3. Padronizacao (Escalonamento)
# Selecionamos as colunas numericas (SEPALLEN, SEPALWID, PETALLEN, PETALWID)
variaveis_list = ['SEPALLEN', 'SEPALWID', 'PETALLEN', 'PETALWID']
mtz_vlrs = df[variaveis_list].values
species = df['specie'].values

scaler = StandardScaler()
mtz_vlrs_padrozinados = scaler.fit_transform(mtz_vlrs)

print("\nDados padronizados (primeiras 5 linhas):")
print(mtz_vlrs_padrozinados[:5])

# 4. Metodo do Cotovelo (Elbow Method)
print("\n--- Analise do Numero Otimo de Grupos (Metodo do Cotovelo) ---")
ex2_2_clustering_iris.plot_elbow_method(
    mtz_vlrs_padrozinados, 
    max_k=10, 
    filename='kmeans_elbow_iris.png'
)

# 5. Executar K-means
# Vamos utilizar k=3 pois sabemos que existem 3 especies no dataset IRIS
k_ideal = 3
print(f"\n--- Executando K-means (k={k_ideal}) ---")
df_grupos, kmeans_model = ex2_2_clustering_iris.executar_kmeans(
    mtz_vlrs_padrozinados, 
    species, 
    n_clusters=k_ideal
)

# Adicionando a informacao do grupo ao DataFrame original
df['Grupo'] = df_grupos['Grupo']

# 6. Mostrar resultados
print("\n--- Dados Originais com Atribuicao de Grupos (K-means) ---")
print(df.head(10))


# Verificar a contagem em cada grupo
print("\nContagem por grupo:")
print(df_grupos['Grupo'].value_counts())

# Comparar com as especies originais (Crosstab)
print("\n--- Cruzamento: Especie Original vs Grupo K-means ---")
print(pd.crosstab(df['specie'], df['Grupo']))

# 7. Visualizacao dos clusters
print("\n--- Gerando Grafico de Dispersao dos Clusters ---")
ex2_2_clustering_iris.plot_kmeans_clusters(
    mtz_vlrs_padrozinados, 
    df_grupos['Grupo'].values, 
    kmeans_model.cluster_centers_,
    filename='kmeans_scatter_iris.png'
)

print("\nExercicio 2 concluido com sucesso utilizando Pure K-means.")
