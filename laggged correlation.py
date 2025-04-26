import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Load the data
file_path = r"C:\Users\arman\OneDrive\Desktop\AQI proj\Isfahan Air contamination397_404 main time.csv"
df = pd.read_csv(file_path)

# Drop time column
df_values = df.iloc[:, 1:]

# Initialize a dictionary to store correlations for each lag
lag_correlations = {}

# Loop through lags from 0 to 27
for lag in range(28):
    lagged_df = df_values.shift(lag)
    
    # Store correlation matrix for this lag
    corr_matrix = pd.DataFrame(index=df_values.columns, columns=df_values.columns, dtype=float)

    # Compute pairwise Spearman correlation while ignoring pairs with NaNs
    for col1 in df_values.columns:
        for col2 in df_values.columns:
            valid = ~(
                df_values[col1].isna() | 
                lagged_df[col2].isna()
            )
            if valid.sum() > 2:  # Need at least 3 points to compute meaningful correlation
                rho, _ = spearmanr(df_values[col1][valid], lagged_df[col2][valid])
                corr_matrix.loc[col1, col2] = rho
            else:
                corr_matrix.loc[col1, col2] = np.nan
    
    lag_correlations[lag] = corr_matrix

# Example: access correlation matrix at lag 5
lag_5_corr = lag_correlations[5]

# Save all lags into Excel
with pd.ExcelWriter("correlations_lag_0_to_27.xlsx") as writer:
    for lag, matrix in lag_correlations.items():
        matrix.to_excel(writer, sheet_name=f"Lag_{lag}")


# Load Excel file with multiple sheets (Lag_0, Lag_1, ..., Lag_27)
excel_path = "correlations_lag_0_to_27.xlsx"
xls = pd.read_excel(excel_path, sheet_name=None, index_col=0)

# Sort lag sheet names by lag number
lags = sorted(xls.keys(), key=lambda x: int(x.split("_")[1]))

# Collect correlations by (varfix, varlagged) pair
correlation_data = {}
for lag in lags:
    df = xls[lag]
    lag_num = int(lag.split("_")[1])
    for row in df.index:
        for col in df.columns:
            key = (row, col)
            if key not in correlation_data:
                correlation_data[key] = {}
            correlation_data[key][f"lag{lag_num}"] = df.loc[row, col]

# Prepare final rows
records = []
for (varfix, varlagged), lag_dict in correlation_data.items():
    row = {"varfix": varfix, "varlagged": varlagged}
    row.update(lag_dict)

    # Get all lag-values
    sorted_lags = sorted(lag_dict.items(), key=lambda x: -abs(x[1]))  # sort descending by abs value

    # Get first max
    max_lag_1, max_corr_1 = sorted_lags[0]

    # Get second max
    max_lag_2, max_corr_2 = sorted_lags[1]

    # Add new columns
    row["max_lag_1"] = max_lag_1
    row["max_corr_1"] = max_corr_1
    row["max_lag_2"] = max_lag_2
    row["max_corr_2"] = max_corr_2

    records.append(row)

# Build DataFrame
final_df = pd.DataFrame(records)

# Order columns
lag_cols = [f"lag{i}" for i in range(28)]
final_df = final_df[["varfix", "varlagged"] + lag_cols + ["max_lag_1", "max_corr_1", "max_lag_2", "max_corr_2"]]

# Save to Excel
final_df.to_excel("flattened_with_top2_corrs.xlsx", index=False)
