import pandas as pd
from pathlib import Path
import os

def read_data(file_name='tabela_1_presenca_ausencia.csv'):
    # Procura no diretÃ³rio pai por padrao
    path = Path(os.path.dirname(__file__)) / ".." / file_name
    if not path.exists():
        raise FileNotFoundError(f"Arquivo {path} nÃ£o encontrado.")
    return pd.read_csv(path)

if __name__ == "__main__":
    df = read_data()
    print("Dados lidos com sucesso:")
    print(df.head())
