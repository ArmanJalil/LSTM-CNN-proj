import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
from persiantools.jdatetime import JalaliDateTime
from matplotlib.dates import DateFormatter

# Load the CSV file
file_path = r"C:\Users\arman\OneDrive\Desktop\AQI proj\Isfahan Air contamination397_404 main time.csv"
df = pd.read_csv(file_path)

# Print column names to debug
print("Column names:", df.columns.tolist())

# Assuming the first column contains the dates
date_col = df.columns[0]

# Convert Persian (Jalali) dates to Gregorian dates
def persian_to_gregorian(persian_date_str):
    try:
        persian_date = JalaliDateTime.strptime(persian_date_str, '%Y/%m/%d %H:%M:%S')
        gregorian_date = persian_date.to_gregorian()
        return gregorian_date
    except Exception as e:
        print(f"Error parsing date {persian_date_str}: {e}")
        return pd.NaT

# Apply the conversion to the date column
df['Gregorian_Date'] = df[date_col].apply(persian_to_gregorian)
df = df.dropna(subset=['Gregorian_Date'])

# Extract data for the month of May
df_may = df[df['Gregorian_Date'].dt.month == 5].copy()

# Define pollutant columns (excluding the date columns)
pollutant_cols = df.columns[1:-1]

# Drop columns with all NaN or constant values
valid_cols = [col for col in pollutant_cols if df_may[col].nunique() > 1 and df_may[col].notna().sum() > 0]
df_may = df_may[['Gregorian_Date'] + valid_cols]

# Smooth the data with a rolling mean (window of 5 to further reduce short-term spikes)
for col in valid_cols:
    df_may[col] = df_may[col].rolling(window=5, min_periods=1, center=True).mean()

# Initialize a list to store change point results
change_points_data = []

# Split variables into groups of 5 for better readability
vars_per_plot = 5
num_plots = (len(valid_cols) + vars_per_plot - 1) // vars_per_plot  # Ceiling division

for plot_idx in range(num_plots):
    start_idx = plot_idx * vars_per_plot
    end_idx = min((plot_idx + 1) * vars_per_plot, len(valid_cols))
    current_cols = valid_cols[start_idx:end_idx]
    num_current_vars = len(current_cols)
    
    # Set up figure for this group
    figsize = (15, 4 * num_current_vars)  # 4 inches per subplot
    plt.figure(figsize=figsize)
    
    # Loop through the current group of pollutants
    for idx, col in enumerate(current_cols, 1):
        # Prepare the time-series data (ignore NaN values)
        series = df_may[['Gregorian_Date', col]].dropna()
        if len(series) < 10:
            continue
        
        # Filter out periods where the value is effectively zero (e.g., < 0.1)
        series = series[series[col] > 0.1]
        if len(series) < 10:
            continue
        
        signal = series[col].values
        dates = series['Gregorian_Date'].values
        
        # Apply change point detection using Pelt search method
        model = "l2"
        algo = rpt.Pelt(model=model, min_size=3, jump=1).fit(signal)
        change_points = algo.predict(pen=100)  # Increased penalty to 100 for fewer change points
        
        # Map change points to dates
        change_indices = change_points[:-1]  # Exclude the last point (end of series)
        
        # Filter change points based on the magnitude of the mean shift
        filtered_change_indices = []
        for i in range(len(change_indices)):
            if i == 0:
                prev_mean = np.mean(signal[:change_indices[i]])
            else:
                prev_mean = np.mean(signal[change_indices[i-1]:change_indices[i]])
            next_mean = np.mean(signal[change_indices[i]:change_points[i+1] if i+1 < len(change_points) else None])
            mean_diff = abs(next_mean - prev_mean)
            # Only keep change points where the mean shift is significant (e.g., > 20% of the overall mean)
            if mean_diff > 0.2 * np.mean(signal):
                filtered_change_indices.append(change_indices[i])
        
        change_dates = dates[filtered_change_indices]
        
        # Store results
        for cp_date in change_dates:
            value_at_cp = series.loc[series['Gregorian_Date'] == cp_date, col].values[0]
            change_points_data.append({
                'Pollutant': col,
                'Change Point Date': cp_date,
                'Value at Change Point': value_at_cp
            })
        
        # Plot the time-series with change points
        plt.subplot(num_current_vars, 1, idx)
        plt.plot(dates, signal, label=col, color='blue')
        for cp_date in change_dates:
            plt.axvline(x=cp_date, color='red', linestyle='--')  # Removed label to avoid legend clutter
        plt.xlabel('Date', fontsize=10)
        plt.ylabel(col, fontsize=10)
        plt.title(f'Time-Series of {col} in May with Change Points', fontsize=10)
        plt.grid(True)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        # Format x-axis dates to show fewer ticks and avoid overlap
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit to 10 date ticks
    
    # Save the plot for this group
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Increased vertical spacing
    plt.savefig(f"may_pollutant_shifts_part_{plot_idx + 1}.png", dpi=100, bbox_inches='tight')
    plt.close()

# Save change points to CSV
change_points_df = pd.DataFrame(change_points_data)
change_points_df.to_csv("may_change_points.csv", index=False)

print("Change points plots saved as 'may_pollutant_shifts_part_1.png', 'may_pollutant_shifts_part_2.png', etc.")
print("Change points data saved as 'may_change_points.csv'")
