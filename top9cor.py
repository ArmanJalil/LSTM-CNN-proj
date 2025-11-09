import pandas as pd
import os

# Your existing code
file_path = r"C:\Users\arman\OneDrive\Desktop\AQIorgonized\merged_pollution_weather_with_time.csv"
df = pd.read_csv(file_path).iloc[:, 1:]
corr_matrix = df.corr(method='spearman')

# Calculate NaN percentages for each column
nan_percentages = (df.isnull().sum() / len(df)) * 100
nan_percentages = nan_percentages.round(2)

# Create a mapping of column names to their numbers
column_numbers = {col: idx for idx, col in enumerate(df.columns, 1)}

# Create top9cor dictionary with numbered structure and NaN info
top9cor = {}

for column in corr_matrix.columns:
    # Get correlations for current column, exclude self-correlation
    corr_series = corr_matrix[column].drop(column)
    
    # Get top 9 correlations (absolute value to get strongest relationships)
    top_correlations = corr_series.abs().sort_values(ascending=False).head(9)
    
    # Get the actual correlation values with column numbers
    top_cor_dict = {}
    for var in top_correlations.index:
        top_cor_dict[f"{column_numbers[var]}. {var}"] = {
            'correlation': corr_series[var],
            'nan_percentage': nan_percentages[var]
        }
    
    top9cor[f"{column_numbers[column]}. {column}"] = {
        'nan_percentage': nan_percentages[column],
        'top_correlations': top_cor_dict
    }

# Print the results with NaN percentages
print("TOP 9 CORRELATIONS WITH NaN PERCENTAGES")
print("=" * 60)
for variable, data in top9cor.items():
    nan_pct = data['nan_percentage']
    correlations = data['top_correlations']
    
    print(f"\n{variable} [NaN: {nan_pct}%]:")
    for correlated_var, corr_data in correlations.items():
        corr_value = corr_data['correlation']
        var_nan_pct = corr_data['nan_percentage']
        print(f"  {correlated_var} [NaN: {var_nan_pct}%]: {corr_value:.4f}")

# CORRECTED CODE: Create dictionary with ALL columns as targets, but only use correlated variables with <24% NaN
filtered_top_cor = {}

for column in corr_matrix.columns:
    # Get correlations for current column, exclude self-correlation
    corr_series = corr_matrix[column].drop(column)
    
    # Get top correlations (absolute value) and filter for variables with <24% NaN
    all_correlations = corr_series.abs().sort_values(ascending=False)
    
    # Filter to only include correlated variables with <24% NaN
    filtered_correlations = {}
    for var in all_correlations.index:
        if nan_percentages[var] < 24:  # Only include if correlated variable has <24% NaN
            filtered_correlations[var] = {
                'correlation': corr_series[var],
                'nan_percentage': nan_percentages[var]
            }
    
    # Take top 9 from filtered results
    top_filtered = dict(list(filtered_correlations.items())[:9])
    
    # Extract just the column numbers for the final output
    correlated_numbers = [column_numbers[var] for var in top_filtered.keys()]
    
    # Add to dictionary - ALL columns are included as targets, regardless of their NaN percentage
    filtered_top_cor[column_numbers[column]] = correlated_numbers

print("\n" + "=" * 60)
print("FILTERED CORRELATIONS (ALL targets, correlated features with <24% NaN)")
print("=" * 60)
print(f"\nDictionary format:")
print(filtered_top_cor)

print(f"\nDetailed view:")
for col_num, correlated_nums in filtered_top_cor.items():
    target_nan_pct = nan_percentages[[name for name, num in column_numbers.items() if num == col_num][0]]
    print(f"Column {col_num} [NaN: {target_nan_pct}%]: {correlated_nums}")

# Export to CSV
def export_to_csv(top9cor, filename="top9_correlations_with_nan.csv"):
    """Export the correlation data to a CSV file"""
    export_data = []
    
    for main_var, data in top9cor.items():
        main_nan_pct = data['nan_percentage']
        correlations = data['top_correlations']
        
        for corr_var, corr_data in correlations.items():
            export_data.append({
                'Main_Variable': main_var,
                'Main_Variable_NaN_Percentage': main_nan_pct,
                'Correlated_Variable': corr_var,
                'Correlated_Variable_NaN_Percentage': corr_data['nan_percentage'],
                'Correlation_Value': corr_data['correlation'],
                'Absolute_Correlation': abs(corr_data['correlation'])
            })
    
    export_df = pd.DataFrame(export_data)
    export_df.to_csv(filename, index=False, encoding='utf-8-sig')
    return export_df

# Export the filtered correlations as well
def export_filtered_correlations(filtered_top_cor, filename="filtered_correlations_less_24_nan.csv"):
    """Export the filtered correlation data to a CSV file"""
    filtered_data = []
    
    for main_col_num, correlated_nums in filtered_top_cor.items():
        # Get the actual column name
        main_col_name = [name for name, num in column_numbers.items() if num == main_col_num][0]
        
        filtered_data.append({
            'Column_Number': main_col_num,
            'Column_Name': main_col_name,
            'NaN_Percentage': nan_percentages[main_col_name],
            'Top_Correlated_Column_Numbers': str(correlated_nums),
            'Number_of_Correlations': len(correlated_nums)
        })
    
    filtered_df = pd.DataFrame(filtered_data)
    filtered_df.to_csv(filename, index=False, encoding='utf-8-sig')
    return filtered_df

# Export the data
export_df = export_to_csv(top9cor)
print(f"\nData exported to 'top9_correlations_with_nan.csv'")
print(f"Exported {len(export_df)} rows of correlation data")

# Export filtered correlations
filtered_export_df = export_filtered_correlations(filtered_top_cor)
print(f"Filtered correlations exported to 'filtered_correlations_less_24_nan.csv'")
print(f"Exported {len(filtered_export_df)} rows of filtered correlation data")

# Also create a summary CSV with just the variable info
summary_data = []
for variable, data in top9cor.items():
    summary_data.append({
        'Variable_Number_Name': variable,
        'NaN_Percentage': data['nan_percentage']
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("variable_nan_summary.csv", index=False, encoding='utf-8-sig')
print("Variable summary exported to 'variable_nan_summary.csv'")

# Display first few rows of exported data
print("\nFirst 5 rows of exported correlation data:")
print(export_df.head())

# Display filtered dictionary
print(f"\nFiltered correlations dictionary (ALL targets, correlated features with <24% NaN):")
print("filtered_top_cor =", filtered_top_cor)