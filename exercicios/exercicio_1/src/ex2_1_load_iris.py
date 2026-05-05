import pandas as pd
import os
from pathlib import Path

def load_iris_data(file_path=None):
    """
    Loads the IRIS_2023.xls dataset.
    """
    if file_path is None:
        # Default path relative to src/
        file_path = Path(__file__).parent / ".." / ".." / "IRIS_2023.xls"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    # Using xlrd engine as seen in previous research
    df = pd.read_excel(file_path, engine='xlrd')
    return df

if __name__ == "__main__":
    try:
        df = load_iris_data()
        print("IRIS Dataset loaded successfully:")
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error: {e}")
