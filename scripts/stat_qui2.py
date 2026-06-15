import pandas as pd
import sys
import os
import numpy as np

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

def calculate_chi2_step_by_step(df):
    """Calculates Chi-squared expected values step-by-step."""
    # Identify numeric columns (the contingency table)
    # We assume the first column might be a label (like 'Grupo')
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        print("No numerical data found to perform Chi-squared analysis.")
        return

    print("Step 1: Observed Values (O)")
    print(numeric_df)
    
    # Step 2: Calculate Marginals
    row_totals = numeric_df.sum(axis=1)
    col_totals = numeric_df.sum(axis=0)
    grand_total = numeric_df.values.sum()
    
    print("\nStep 2: Marginal Totals")
    print(f"Row Totals:\n{row_totals}")
    print(f"\nColumn Totals:\n{col_totals}")
    print(f"\nGrand Total: {grand_total}")
    
    # Step 3: Calculate Expected Values (E)
    # E_ij = (Row Total_i * Col Total_j) / Grand Total
    expected = np.outer(row_totals, col_totals) / grand_total
    expected_df = pd.DataFrame(expected, columns=numeric_df.columns, index=numeric_df.index)
    
    print("\nStep 3: Expected Values (E)")
    print("Formula: E_ij = (Row Total_i * Column Total_j) / Grand Total")
    print(expected_df)
    
    # Step 4: Combined View (O and E)
    print("\nStep 4: Combined View [Observed (Expected)]")
    combined_df = numeric_df.copy().astype(str)
    for row in numeric_df.index:
        for col in numeric_df.columns:
            o = numeric_df.loc[row, col]
            e = expected_df.loc[row, col]
            combined_df.loc[row, col] = f"{o} ({e:.2f})"
    print(combined_df)
    
    # Step 5: Chi-squared contribution (O-E)^2 / E
    chi2_contrib = ((numeric_df - expected_df)**2) / expected_df
    print("\nStep 5: Chi-squared Contribution ((O-E)^2 / E)")
    print(chi2_contrib)
    
    chi2_stat = chi2_contrib.values.sum()
    print(f"\nChi-squared Statistic (Σ): {chi2_stat:.4f}")

def main():
    if len(sys.argv) < 3:
        print("Usage: ./venv/bin/python3 scripts/stat_qui2.py --show-expected-values <filename>")
        sys.exit(1)

    flag = sys.argv[1]
    file_path = sys.argv[2]

    if flag != "--show-expected-values":
        print(f"Unknown flag: {flag}")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    df = load_data(file_path)
    if df is not None:
        calculate_chi2_step_by_step(df)

if __name__ == "__main__":
    main()
