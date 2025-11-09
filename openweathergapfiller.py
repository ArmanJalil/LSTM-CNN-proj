import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# --- Load data ---
pca_df = pd.read_csv(r"C:\Users\arman\OneDrive\Desktop\AQIorgonized\pollution_pca_12components.csv", 
                     index_col=0, parse_dates=True)

# --- Reindex hourly ---
full_range = pd.date_range(pca_df.index.min(), pca_df.index.max(), freq='H')
pca_full = pca_df.reindex(full_range)
print(f"Missing hours: {len(full_range) - len(pca_df)}")

# --- Step 1: Fill completely missing days using hourly mean pattern ---
hourly_pattern = pca_full.groupby(pca_full.index.hour).mean()
all_days = pd.date_range(pca_full.index.min().normalize(), pca_full.index.max().normalize(), freq='D')
existing_days = pca_full.dropna(how='all').index.normalize().unique()
missing_days = sorted(set(all_days) - set(existing_days))

for day in missing_days:
    pca_full.loc[pd.date_range(day, day + pd.Timedelta(hours=23), freq='H')] = hourly_pattern.values

print(f"Filled {len(missing_days)} fully-missing days using hourly averages.")

# --- Step 2: Time features for model-based imputation ---
pca_full['hour_sin'] = np.sin(2*np.pi*pca_full.index.hour/24)
pca_full['hour_cos'] = np.cos(2*np.pi*pca_full.index.hour/24)
pca_full['doy_sin'] = np.sin(2*np.pi*pca_full.index.dayofyear/365)
pca_full['doy_cos'] = np.cos(2*np.pi*pca_full.index.dayofyear/365)

# --- Apply Random Forest-based iterative imputation ---
pca_cols = [f'Pollution_PC{i+1}' for i in range(12)]
feat_cols = ['hour_sin','hour_cos','doy_sin','doy_cos']

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    max_iter=10, random_state=42, initial_strategy='mean'
)
data = pca_full[pca_cols + feat_cols]
imputed = imputer.fit_transform(data)

# --- Build final DataFrame ---
final_df = pd.DataFrame(imputed, index=pca_full.index, columns=pca_cols + feat_cols)
final_df = final_df[pca_cols]

# --- Save result ---
final_df.to_csv(r"C:\Users\arman\OneDrive\Desktop\AQIorgonized\pollution_pca_12components_imputed.csv")
print("✅ All gaps filled and saved successfully!")
