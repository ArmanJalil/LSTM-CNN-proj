import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r"C:\Users\arman\OneDrive\Desktop\AQI proj\Isfahan Air contamination397_404 main time.csv"
df = pd.read_csv(file_path)

# Define pollutant columns (excluding Date)
pollutant_cols = df.columns[1:]  # Exclude 'Date'

# Drop columns with all NaN or constant values (to avoid invalid plots)
valid_cols = [col for col in pollutant_cols if df[col].nunique() > 1 and df[col].notna().sum() > 0]
df = df[valid_cols]

# Set Seaborn style for better readability
sns.set(style="ticks")

# Compute the number of variables to set figure size
num_vars = len(valid_cols)

# Create a large figure size to accommodate all variables
# Each subplot needs about 0.5 inches per variable for visibility
figsize = (max(20, num_vars * 0.5), max(20, num_vars * 0.5))

# Create the pair plot (scatter plot matrix)
# - diag_kind='hist' to show histograms on the diagonal
# - plot_kws adjusts the scatter plot appearance (e.g., smaller points, transparency)
pair_plot = sns.pairplot(
    df,
    diag_kind='hist',
    plot_kws={'alpha': 0.5, 's': 10},  # Transparency and smaller point size for readability
    height=2.5,  # Height of each subplot
)

# Adjust the layout to prevent overlap
pair_plot.figure.suptitle("Scatter Plot Matrix of AQI Variables", y=1.02, fontsize=16)
pair_plot.figure.set_size_inches(figsize)

# Rotate the variable names (axis labels) instead of tick labels
for i, j in zip(*plt.np.triu_indices_from(pair_plot.axes, k=1)):
    # Lower triangle (scatter plots below the diagonal)
    if i > j:
        pair_plot.axes[i, j].set_xlabel(pair_plot.axes[i, j].get_xlabel(), rotation=90, fontsize=max(6, 12 - num_vars * 0.1))
        pair_plot.axes[i, j].set_ylabel(pair_plot.axes[i, j].get_ylabel(), rotation=0, fontsize=max(6, 12 - num_vars * 0.1))
    # Upper triangle (scatter plots above the diagonal)
    elif i < j:
        pair_plot.axes[i, j].set_xlabel(pair_plot.axes[i, j].get_xlabel(), rotation=90, fontsize=max(6, 12 - num_vars * 0.1))
        pair_plot.axes[i, j].set_ylabel(pair_plot.axes[i, j].get_ylabel(), rotation=0, fontsize=max(6, 12 - num_vars * 0.1))

# Rotate x-axis labels for the diagonal histograms (bottom row)
for j in range(num_vars):
    pair_plot.axes[num_vars-1, j].set_xlabel(pair_plot.axes[num_vars-1, j].get_xlabel(), rotation=90, fontsize=max(6, 12 - num_vars * 0.1))

# Rotate y-axis labels for the diagonal histograms (leftmost column)
for i in range(num_vars):
    pair_plot.axes[i, 0].set_ylabel(pair_plot.axes[i, 0].get_ylabel(), rotation=0, fontsize=max(6, 12 - num_vars * 0.1))

# Adjust tick labels (numerical values) to be readable but not rotated
for ax in pair_plot.axes.flatten():
    ax.tick_params(axis='x', labelrotation=0, labelsize=max(6, 12 - num_vars * 0.1))
    ax.tick_params(axis='y', labelrotation=0, labelsize=max(6, 12 - num_vars * 0.1))

# Save the plot with high DPI for zooming
pair_plot.figure.savefig("scatter_plot_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

print("Scatter plot matrix saved as 'scatter_plot_matrix.png'")