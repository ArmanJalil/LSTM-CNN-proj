import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy.stats as stats

# Load data and drop index
file_path = r"C:\Users\arman\OneDrive\Desktop\AQIorgonized\merged_pollution_weather.csv"
df = pd.read_csv(file_path).iloc[:, 1:]
cols = df.columns
n = len(cols)

# Spearman correlation
corr_matrix, _ = stats.spearmanr(df)
corr_matrix = corr_matrix[:n, :n]

# Create lower triangle correlation matrix
combined_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i < j:
            combined_matrix[i, j] = np.nan
        elif i > j:
            combined_matrix[i, j] = corr_matrix[i, j]
        else:
            combined_matrix[i, j] = 0

# Color map and normalization
cmap = mcolors.LinearSegmentedColormap.from_list("blue_white_red", ["blue", "white", "red"])
norm = mcolors.Normalize(vmin=-1, vmax=1)

# Double-large figure
fig, axes = plt.subplots(n, n, figsize=(n * 1.6, n * 1.6))
fig.subplots_adjust(wspace=0.1, hspace=0.1)

# Fill cells
for i in range(n):
    for j in range(n):
        ax = axes[i, j]

        if i == j:
            ax.axis('off')
        elif i < j:
            ax.scatter(df.iloc[:, j], df.iloc[:, i], s=2)
        else:
            ax.imshow([[combined_matrix[i, j]]], cmap=cmap, norm=norm)

        ax.set_xticks([])
        ax.set_yticks([])

        # Labels
        if j == 0:
            ax.set_ylabel(cols[i], rotation=45, fontsize=40, fontweight='bold', labelpad=50, va='bottom')
        if i == n - 1:
            ax.set_xlabel(cols[j], rotation=45, fontsize=40, fontweight='bold', labelpad=50, ha='right')

# Colorbar
cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
cb = fig.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cb.set_label("Spearman Correlation", fontsize=13, fontweight='bold')

# Title and layout
fig.suptitle("Scatter Plot + Spearman Correlation Matrix", fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()
