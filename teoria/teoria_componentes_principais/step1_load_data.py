import sys
import os

# Add the libs directory to the path to promote reuse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../exercicios/libs')))

from data_io.data_loader import load_pca_data, show_head

def main():
    csv_path = os.path.join(os.path.dirname(__file__), 'exemplo_pca.csv')
    print(f"Loading data from: {csv_path}")
    
    try:
        data = load_pca_data(csv_path)
        print("\nFirst lines of the PCA example data:")
        show_head(data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
