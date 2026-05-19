import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Adicionando o caminho das libs para reuso
LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../libs'))

if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)

from data_io.data_loader import load_pca_data

def main():
    # Configuracoes de exibicao do Pandas
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    print("=== Passo 1: Carregamento de Dados ===")
    csv_path = os.path.join(os.path.dirname(__file__), 'exemplo_pca.csv')
    try:
        data = load_pca_data(csv_path)
        print("Dados originais (primeiras linhas):")
        print(data.values)
        
        X = data.values
        
        print("\n=== Passo 2: Padronizacao (Z-score) ===")
        # Padronizacao: (X - media) / desvio_padrao
        # ddof=1 para desvio padrao amostral (corrigido)
        mean_X = np.mean(X, axis=0)

        print("\n=== Passo 2.1: Media de cada coluna ===")
        for i, col in enumerate(data.columns):
            print(f"{col}: {mean_X[i]:.4f}")

        #ddof degree of freedom
        #para amostras usamos 1
        std_X = np.std(X, axis=0, ddof=1)
        X_std = (X - mean_X) / std_X
        
        print("Dados Padronizados (X_std):")
        print(pd.DataFrame(X_std, columns=data.columns).values)

        print("\n=== Passo 3: Matriz de Covariancia, Autovalores e Autovetores ===")
        # A matriz de covariancia de dados padronizados e a matriz de correlacao
        cov_mat = np.cov(X_std.T)
        print("Matriz de Covariancia (R):")
        print(cov_mat)

        # Calculo de Autovalores e Autovetores
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)



        # Ordenar de forma decrescente pelos autovalores (Variancia)
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]




        print("\nAutovalores (Variancia):")
        print(eig_vals)

        print("\nAutovetores (Cargas dos Componentes):")
        print(eig_vecs)



        print("\n=== Passo 4: Valores Originais com os Escores dos Individuos ===")
        # O calculo dos escores e o produto matricial dos dados padronizados pelos autovetores
        scores = np.dot(X_std, eig_vecs)
        
        # Criando o DataFrame dos escores organizados (com index iniciando em 1)
        component_names = [f'S_CP{i+1}' for i in range(len(eig_vals))]
        df_scores = pd.DataFrame(scores, columns=component_names, index=data.index + 1)
        
        # Ajustando o index dos dados originais para iniciar em 1 para alinhar a juncao
        df_original = data.copy()
        df_original.index = data.index + 1
        
        # Concatenando os dados originais com os escores lado a lado (eixo 1)
        df_final = pd.concat([df_original, df_scores], axis=1)
        df_final.index.name = 'unidade'
        
        print("Tabela Completa (Dados Originais + Escores):")
        print(df_final)






        
        # Variancia Explicada
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
         
        print("\nVariancia Explicada Acumulada:")
        cum_var_exp = np.cumsum(var_exp)
        for i, (v, cv) in enumerate(zip(var_exp, cum_var_exp)):
            print(f"CP{i+1}: Individual = {v:.2f}% | Acumulada = {cv:.2f}%")




        print("\n=== Passo 5: Geracao dos Graficos ===")
        # Definir caminho para salvar as imagens na teoria
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../teoria/teoria_componentes_principais/imgs'))
        os.makedirs(output_dir, exist_ok=True)

        # -------------------------------------------------------------
        # GRAFICO 1: Scree Plot (Grafico de Autovalores)
        # -------------------------------------------------------------
        plt.figure(figsize=(8, 5))
        # Variancia explicada em percentual
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in eig_vals]
        
        plt.plot(range(1, len(eig_vals) + 1), eig_vals, marker='o', linestyle='-', color='blue')
        plt.axhline(y=1.0, color='red', linestyle='--', label='Criterio de Kaiser (λ = 1)')
        
        # Adicionando os rotulos das porcentagens sobre os pontos
        for i, val in enumerate(var_exp):
            plt.text(i + 1, eig_vals[i] + 0.05, f"{val:.2f}%", ha='center', va='bottom', fontweight='bold')
            
        plt.title('Grafico Scree Plot')
        plt.xlabel('Numero do Autovalor (Componente)')
        plt.ylabel('Autovalor (Eigenvalue)')
        plt.xticks(range(1, len(eig_vals) + 1))
        plt.ylim(0, max(eig_vals) + 0.4)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        scree_path = os.path.join(output_dir, 'scree_plot.png')
        plt.savefig(scree_path, dpi=300, bbox_inches='tight')
        print(f"Salvo: {scree_path}")
        plt.show()

        # -------------------------------------------------------------
        # GRAFICO 2: Grafico de Escores (CP1 x CP2)
        # -------------------------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        # Plotando os pontos dos individuos (unidades)
        plt.scatter(df_scores['S_CP1'], df_scores['S_CP2'], color='blue', edgecolors='black', s=100, zorder=3)
        
        # Rotulando cada ponto com o numero da unidade correspondente
        for idx, row in df_scores.iterrows():
            plt.text(row['S_CP1'] + 0.05, row['S_CP2'] + 0.05, str(idx), fontsize=12, fontweight='bold', ha='left', va='bottom')
            
        plt.title(f'Grafico CP1 x CP2\n(CP1: {var_exp[0]:.2f}% | CP2: {var_exp[1]:.2f}%)')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal') # Mantem a escala dos eixos proporcional
        
        scores_path = os.path.join(output_dir, 'scores_plot.png')
        plt.savefig(scores_path, dpi=300, bbox_inches='tight')
        print(f"Salvo: {scores_path}")
        plt.show()

        # -------------------------------------------------------------
        # GRAFICO 3: BIPLOT (Escores + Cargas das Variaveis Originais)
        # -------------------------------------------------------------
        plt.figure(figsize=(9, 7))
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        # 1. Plota os escores dos individuos (mesma lógica do gráfico 2)
        plt.scatter(df_scores['S_CP1'], df_scores['S_CP2'], color='blue', edgecolors='black', alpha=0.7, s=80, label='Individuos', zorder=3)
        for idx, row in df_scores.iterrows():
            plt.text(row['S_CP1'] + 0.05, row['S_CP2'] + 0.05, str(idx), fontsize=10, ha='left', va='bottom')
            
        # 2. Plota os vetores das cargas das variaveis originais (multiplicados por um fator de escala para visualizacao)
        # No biplot do slide, as direcoes dependem da correlacao/carga da variavel na CP1 e CP2
        escala_vetor = 2.0  # Fator para esticar os vetores e facilitar a leitura no mesmo grafico
        for i, col_name in enumerate(data.columns):
            # Coordenadas da variavel tiradas das cargas (autovetores) da CP1 e CP2
            # Nota: multiplicamos por np.sqrt(eig_vals) se quisermos correlacoes como no slide
            carga_cp1 = eig_vecs[i, 0] * np.sqrt(eig_vals[0])
            carga_cp2 = eig_vecs[i, 1] * np.sqrt(eig_vals[1])
            
            plt.arrow(0, 0, carga_cp1 * escala_vetor, carga_cp2 * escala_vetor, 
                      color='red', alpha=0.8, head_width=0.08, head_length=0.1, lw=1.5, length_includes_head=True)
            plt.text(carga_cp1 * escala_vetor * 1.15, carga_cp2 * escala_vetor * 1.15, col_name, 
                     color='black', fontsize=12, fontweight='bold', ha='center', va='center')
            
        plt.title('Grafico BIPLOT')
        plt.xlabel(f'Factor 1: {var_exp[0]:.2f}%')
        plt.ylabel(f'Factor 2: {var_exp[1]:.2f}%')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        
        biplot_path = os.path.join(output_dir, 'biplot.png')
        plt.savefig(biplot_path, dpi=300, bbox_inches='tight')
        print(f"Salvo: {biplot_path}")
        plt.show()


            

            
    except Exception as e:
        print(f"Erro durante a execucao: {e}")

if __name__ == "__main__":
    main()
