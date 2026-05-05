import ex1_1_read_data_from_csv
import ex1_2_clustering_binario
from scipy.spatial.distance import squareform
import pandas as pd

# 1 read data from csv
df = ex1_1_read_data_from_csv.read_data()

# 2 show 10 first rows of the data
print("--- Primeiras 10 linhas do dataframe ---")
print(df.head(10))

# 3 calcula matriz de distancias usando coeficiente Sorensen-Dice e Simple Matching
X = df.iloc[:, 1:].values
species = df.iloc[:, 0].values

# criando e observando as matrizes de distancias
dist_dice = ex1_2_clustering_binario.calcular_matriz_distancias(X, metric='dice')
dist_sm = ex1_2_clustering_binario.calcular_matriz_distancias(X, metric='matching')

# Converter para formato quadrado para visualizacao
# isso porque sem essa conversao as mtz estao no formato otimizado o ndarry
df_dist_dice = pd.DataFrame(squareform(dist_dice), index=species, columns=species)
df_dist_sm = pd.DataFrame(squareform(dist_sm), index=species, columns=species)


print("\n--- Matriz de DistÃ¢ncia (Sorensen-Dice) ---")
print(df_dist_dice.round(3))

print("\n--- Matriz de DistÃ¢ncia (Simple Matching) ---")
print(df_dist_sm.round(3))

# 4 analyze the data using clustering_binario module (gera os dendrogramas)
print("\n--- Gerando Dendrogramas ---")
ex1_2_clustering_binario.plot_dendrogram(dist_dice, species, 'dice', 'Dendrograma - Sorensen-Dice', 'dendrograma_sorensen_dice.png')
ex1_2_clustering_binario.plot_dendrogram(dist_sm, species, 'matching', 'Dendrograma - Simple Matching', 'dendrograma_simple_matching.png')
