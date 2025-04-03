import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Load up
core_path = 'W:/Data/2025_01_20_sensor_temperature_drift/'

all_directories = [os.path.join(core_path, folder) for folder in os.listdir(core_path)
                   if os.path.isdir(os.path.join(core_path, folder))]

folder_data_pairs = []

# Loop through each directory and load the files
for directory in all_directories:
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    header_file = None
    sample_file = None
    
    # Separate header and sample files within each directory
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        if 'header' in csv_file:
            header_file = pd.read_csv(file_path, sep=';')
        else:
            sample_file = pd.read_csv(file_path, sep=';')
    
    # Store the pair of header and sample files along with the directory name
    if header_file is not None and sample_file is not None:
        folder_name = os.path.basename(directory)  # Get the name of the folder
        folder_data_pairs.append((header_file, sample_file, folder_name))

# Initialize the figure for 3 subplots
plt.figure(figsize=(12, 12))

ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

# Loop through each folder, compute correlations and plot
for header_file, sample_file, folder_name in folder_data_pairs:
    # Extract relevant data
    x_ind = sample_file.index[sample_file['fieldname'] == 'x'].tolist()
    x_data = sample_file.iloc[x_ind, 4:-1].to_numpy()

    laser_power_ind = sample_file.index[sample_file['fieldname'] == 'auxin0'].tolist()
    laser_power_data = sample_file.iloc[laser_power_ind, 4:-1].to_numpy()

    laser_power_avg = np.mean(laser_power_data, 1)
    x_normalised = x_data - x_data.mean(axis=1, keepdims=True)
    amplitude = np.max(x_normalised, axis=1)
    runs = np.arange(0, amplitude.shape[0])

    # Remove NaN values
    valid_indices = ~np.isnan(laser_power_avg) & ~np.isnan(amplitude)
    laser_power_avg_clean = laser_power_avg[valid_indices]
    amplitude_clean = amplitude[valid_indices]
    runs_clean = runs[valid_indices]

    # Compute Pearson correlation for this pair of files
    if len(laser_power_avg_clean) > 1:  # Avoid calculation if there's insufficient data
        corr_coeff, _ = pearsonr(laser_power_avg_clean, amplitude_clean)
    else:
        corr_coeff = np.nan  # Not enough data for correlation

    # Fit a local linear regression model
    reg = LinearRegression()
    reg.fit(laser_power_avg_clean.reshape(-1, 1), amplitude_clean)
    amplitude_pred = reg.predict(laser_power_avg_clean.reshape(-1, 1))
    residuals = amplitude_clean - amplitude_pred

    # Plot data for this folder
    ax1.plot(runs_clean, amplitude_clean, marker='o', label=f'{folder_name} (r={corr_coeff:.2f})')
    ax2.plot(runs_clean, laser_power_avg_clean, marker='o', label=f'{folder_name}')
    ax3.plot(runs_clean, residuals, marker='o', label=f'{folder_name}')

# Finalize the first figure
ax1.set_title('Amplitude vs Runs')
ax1.set_xlabel('Runs')
ax1.set_ylabel('Amplitude')
ax1.grid(True)
ax1.legend(loc='upper right')

ax2.set_title('Laser Power Avg vs Runs')
ax2.set_xlabel('Runs')
ax2.set_ylabel('Laser Power Avg')
ax2.grid(True)
ax2.legend(loc='upper right')

ax3.set_title('Residuals vs Runs')
ax3.set_xlabel('Runs')
ax3.set_ylabel('Residuals')
ax3.grid(True)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Create a separate figure for the correlation coefficients
plt.figure(figsize=(8, 6))
correlations = []

# Extract correlation coefficients for visualization
for header_file, sample_file, folder_name in folder_data_pairs:
    x_ind = sample_file.index[sample_file['fieldname'] == 'x'].tolist()
    x_data = sample_file.iloc[x_ind, 4:-1].to_numpy()

    laser_power_ind = sample_file.index[sample_file['fieldname'] == 'auxin0'].tolist()
    laser_power_data = sample_file.iloc[laser_power_ind, 4:-1].to_numpy()

    laser_power_avg = np.mean(laser_power_data, 1)
    x_normalised = x_data - x_data.mean(axis=1, keepdims=True)
    amplitude = np.max(x_normalised, axis=1)

    valid_indices = ~np.isnan(laser_power_avg) & ~np.isnan(amplitude)
    laser_power_avg_clean = laser_power_avg[valid_indices]
    amplitude_clean = amplitude[valid_indices]

    # Compute correlation
    if len(laser_power_avg_clean) > 1:
        corr_coeff, _ = pearsonr(laser_power_avg_clean, amplitude_clean)
    else:
        corr_coeff = np.nan

    correlations.append((folder_name, corr_coeff))

# Sort by folder name for better organization
correlations = sorted(correlations, key=lambda x: x[0])

# Plot correlation coefficients
folder_names = [item[0] for item in correlations]
correlation_values = [item[1] for item in correlations]

plt.barh(folder_names, correlation_values, color='skyblue')
plt.title('Correlation Coefficients per Folder')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Folder')
plt.tight_layout()
plt.show()
