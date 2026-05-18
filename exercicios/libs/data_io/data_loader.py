import pandas as pd
import os

def load_pca_data(file_path):
    """
    Loads PCA example data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df

def show_head(df, n=5):
    """
    Displays the first n rows of a dataframe.
    """
    print(df.head(n))
