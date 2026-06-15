import pandas as pd
import sys
import os

def load_data(file_path):
    """Loads data from CSV or Excel files."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        else:
            print(f"Unsupported file format: {ext}")
            return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/data.py [load|head] <filename>")
        sys.exit(1)

    command = sys.argv[1].lower()
    file_path = sys.argv[2]

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    df = load_data(file_path)

    if df is not None:
        if command == 'load':
            print(f"Successfully loaded {file_path}")
            print(f"Shape: {df.shape}")
        elif command == 'head':
            # Identify numerical columns for marginal totals
            numeric_df = df.select_dtypes(include=['number'])
            
            # Print head
            print("--- Head of Data ---")
            print(df.head())
            
            if not numeric_df.empty:
                print("\n--- Marginal Totals (Numerical Columns) ---")
                # Column totals
                col_totals = numeric_df.sum(axis=0)
                print("Column Totals:")
                print(col_totals)
                
                # Row totals for the head portion (for context)
                print("\nRow Totals (for the head rows):")
                row_totals_head = numeric_df.head().sum(axis=1)
                print(row_totals_head)
                
                # Grand total of numeric data
                print(f"\nGrand Total (All Numerical Data): {numeric_df.values.sum()}")
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)

if __name__ == "__main__":
    main()
