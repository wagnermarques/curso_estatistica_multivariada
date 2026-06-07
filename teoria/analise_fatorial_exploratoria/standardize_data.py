import pandas as pd
import os

# Determina o diretório onde o script está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))

# Caminhos dos arquivos baseados na localização do script
input_file = os.path.join(script_dir, '../../quimica_do_solo.xlsx')
output_file = os.path.join(script_dir, 'dados_padronizados.csv')

# Normaliza os caminhos para evitar problemas com ../
input_file = os.path.normpath(input_file)
output_file = os.path.normpath(output_file)

if not os.path.exists(input_file):
    print(f"Erro: Arquivo {input_file} não encontrado.")
else:
    # Carregamento dos dados
    cols = ['Ca', 'Mg', 'H + Al', 'Al3+', 'S', 'Fe', 'Mn']
    df = pd.read_excel(input_file)
    X = df[cols]

    # Cálculo da padronização (Z-score)
    df_std = (X - X.mean()) / X.std()

    # Salvar em CSV
    df_std.to_csv(output_file, index=False)
    print(f"Dados padronizados salvos em: {output_file}")
    print("\nPrimeiras linhas dos dados padronizados:")
    print(df_std.head())
