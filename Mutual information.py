import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from scipy.cluster import hierarchy

# Load the CSV file
file_path = r"C:\Users\arman\OneDrive\Desktop\AQI proj\Isfahan Air contamination397_404 main time.csv"
df = pd.read_csv(file_path)

# Define pollutant columns (excluding Date)
pollutant_cols = df.columns[1:]  # Exclude 'Date'

# Drop columns with all NaN or constant values (to avoid invalid MI computations)
valid_cols = [col for col in pollutant_cols if df[col].nunique() > 1 and df[col].notna().sum() > 0]
df = df[valid_cols]

# Function to compute mutual information between two columns, ignoring pairs with NaN
def compute_mi_pairwise(col1, col2):
    # Drop rows where either column is NaN
    mask = ~(col1.isna() | col2.isna())
    if mask.sum() < 2:  # Need at least 2 non-NaN pairs
        return 0
    return mutual_info_regression(col1[mask].values.reshape(-1, 1), col2[mask])[0]

# Compute mutual information matrix
mi_matrix = pd.DataFrame(index=valid_cols, columns=valid_cols, dtype=float)
for col1 in valid_cols:
    for col2 in valid_cols:
        if col1 == col2:
            mi_matrix.loc[col1, col2] = 0
        else:
            mi_matrix.loc[col1, col2] = compute_mi_pairwise(df[col1], df[col2])

# Save MI matrix to CSV
mi_matrix.to_csv("mutual_information_matrix.csv")

# Plot: Dendrogram (tree-style) based on hierarchical clustering of MI
num_vars = len(valid_cols)
plt.figure(figsize=(max(20, num_vars * 0.4), max(10, num_vars * 0.2)))
# Convert MI matrix to distance matrix (1 - normalized MI)
# Normalize MI to [0, 1] by dividing by the maximum MI value
max_mi = mi_matrix.max().max()
if max_mi > 0:
    mi_normalized = mi_matrix / max_mi
else:
    mi_normalized = mi_matrix
distance_matrix = 1 - mi_normalized
# Ensure symmetry and handle any small numerical errors
distance_matrix = (distance_matrix + distance_matrix.T) / 2
np.fill_diagonal(distance_matrix.values, 0)
# Perform hierarchical clustering
linkage = hierarchy.linkage(distance_matrix, method='average')
# Plot dendrogram
dend = hierarchy.dendrogram(linkage, labels=distance_matrix.columns, leaf_rotation=90, leaf_font_size=max(6, 12 - num_vars * 0.1))
plt.title("Hierarchical Clustering Dendrogram of AQI Variables (Mutual Information)", pad=20)
plt.tight_layout()
plt.savefig("mi_dendrogram.png", dpi=300)
plt.close()

print("Mutual information matrix saved as 'mutual_information_matrix.csv'")
print("Dendrogram saved as 'mi_dendrogram.png'")