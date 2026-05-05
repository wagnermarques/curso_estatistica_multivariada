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
# Selecionamos as colunas numéricas (SEPALLEN, SEPALWID, PETALLEN, PETALWID)
features = ['SEPALLEN', 'SEPALWID', 'PETALLEN', 'PETALWID']
X = df[features].values
species = df['specie'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nDados padronizados (primeiras 5 linhas):")
print(X_scaled[:5])

# 4. Produce distance matrix (Euclidean)
dist_euclidean = pdist(X_scaled, metric='euclidean')

# 5. Show preview of distance matrix (squareform)
# Mostrando apenas um subconjunto 5x5 para não poluir o terminal
df_dist = pd.DataFrame(squareform(dist_euclidean))
print("\n--- Prévia da Matriz de Distância (Subconjunto 5x5) ---")
print(df_dist.iloc[:5, :5].round(3))

# 6. Produce the Dendrogram
print("\n--- Gerando Dendrograma Hierárquico (IRIS) ---")
ex2_2_clustering_iris.plot_dendrogram_iris(
    dist_euclidean, 
    labels=species, 
    title='Dendrograma IRIS - Distância Euclidiana (Ward)', 
    filename='dendrograma_iris_ward.png'
)
