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

# 8. Analise Univariada (ANOVA) - Verificando a qualidade dos grupos
import statsmodels.api as sm
from statsmodels.formula.api import ols

print("\n--- Analise de Variancia (ANOVA) por Variavel ---")
print("H0: Nao ha diferenca entre as medias dos grupos")
print("H1: Ha pelo menos uma diferenca entre as medias")

for var in variaveis_list:
    formula = f"{var} ~ C(Grupo)"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Acessando os valores usando o nome da linha 'C(Grupo)'
    p_valor = anova_table.at['C(Grupo)', 'PR(>F)']
    f_stat = anova_table.at['C(Grupo)', 'F']
    
    print(f"\nVariavel: {var}")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-valor: {p_valor:.4e}")
    
    if p_valor < 0.05:
        print("Resultado: Significativo (Existem diferencas entre os grupos)")
    else:
        print("Resultado: Nao significativo (Nao ha diferencas claras)")

# 9. Analise Multivariada de Variancia (MANOVA)
from statsmodels.multivariate.manova import MANOVA

print("\n--- Analise Multivariada de Variancia (MANOVA) ---")
print("H0: Os vetores de medias dos grupos sao iguais")
print("H1: Pelo menos um vetor de media e diferente")

# Criando a formula para todas as variaveis dependentes
formula_manova = " + ".join(variaveis_list) + " ~ C(Grupo)"
manova = MANOVA.from_formula(formula_manova, data=df)

# Exibindo os resultados (Wilks' lambda, Pillai's trace, etc)
print(manova.mv_test())

