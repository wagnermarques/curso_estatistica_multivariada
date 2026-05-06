import ex2_1_load_iris
import ex2_2_clustering_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# 1. Read data
df = ex2_1_load_iris.load_iris_data()

# 2. Show first 10 rows
print("--- Primeiras 10 linhas do dataset IRIS ---")
print(df.head(10))

# 3. Standardization (Scale)
# Selecionamos as colunas numericas (SEPALLEN, SEPALWID, PETALLEN, PETALWID)
variaveis_list = ['SEPALLEN', 'SEPALWID', 'PETALLEN', 'PETALWID']
mtz_vlrs = df[variaveis_list].values
species = df['specie'].values

print("Matriz de valores (primeiras 5 linhas):")
print(mtz_vlrs[:5])

scaler = StandardScaler()
mtz_vlrs_padrozinados = scaler.fit_transform(mtz_vlrs)

print("\nDados padronizados (primeiras 5 linhas):")
print(mtz_vlrs_padrozinados[:5])

# 4. Produce distance matrix (Euclidean)
dist_euclidean = pdist(mtz_vlrs_padrozinados, metric='euclidean')

# 5. Show preview of distance matrix (squareform)
# Mostrando apenas um subconjunto 5x5 para nÃ£o poluir o terminal
df_dist = pd.DataFrame(squareform(dist_euclidean))
print("\n--- Previa da Matriz de Distancia (Subconjunto 5x5) ---")
print(df_dist.iloc[:5, :5].round(3))

# 6. Produce the Dendrogram
print("\n--- Gerando Dendrograma Hierarquico (IRIS) ---")
ex2_2_clustering_iris.plot_dendrogram_iris(
    dist_euclidean, 
    labels=species, 
    title='Dendrograma IRIS - Distancia Euclidiana (Ward)', 
    filename='dendrograma_iris_ward.png'
)


# 7. Cluster Assignment
# Vamos atribuir 3 grupos (k=3) pois sabemos que existem 3 especies
df_grupos = ex2_2_clustering_iris.atribuir_grupos(dist_euclidean, species, t=3, criterion='maxclust')
print("\n--- Atribuicao de Grupos (Ward, k=3) ---")
print(df_grupos.head(10))

# Verificar a contagem em cada grupo
print("\nContagem por grupo:")
print(df_grupos['Grupo'].value_counts())

# Opcional: Comparar com as especies originais (Crosstab)
print("\n--- Cruzamento: Especie Original vs Grupo Hierarquico ---")
print(pd.crosstab(df_grupos['Especie'], df_grupos['Grupo']))

