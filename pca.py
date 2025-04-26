# Importing required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv(r"C:\Users\arman\OneDrive\Desktop\AQI proj\dataforpca.csv")

# Separating features from the target variable if necessary
# Assuming the last column is the target variable
X = data.iloc[:, :-1].values  # Features
# y = data.iloc[:, -1].values  # Uncomment if you have a target variable

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Creating a DataFrame with the PCA results
pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(10)])

# Optionally, you can save the PCA results to a new CSV file
pca_df.to_csv(r"C:\Users\arman\OneDrive\Desktop\AQI proj\dataforpca_reduced.csv", index=False)

# Display the explained variance ratio
print(pca.explained_variance_ratio_)
