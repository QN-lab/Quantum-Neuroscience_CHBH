
#%%

import os
import pandas as pd
import numpy as np

# Directory containing the files
core_path = r'W:\Data\2025_03_06_grad_power\2\spect'

# List all directories inside core_path
all_directories = [
    os.path.join(core_path, folder)
    for folder in os.listdir(core_path)
    if os.path.isdir(os.path.join(core_path, folder))
]
# Sort first by string length, then alphabetically within the same length
all_directories.sort(key=lambda x: (len(os.path.basename(x)), x))

# Target strings to match
target_strings = ['xiy_fft_abs_avg', 'shift_val_fft_abs_avg', 'xiy_fft_abs_filter']

# Dictionary to store extracted data
folder_data = {}

# Loop through each folder
for directory in all_directories:
    # Get all matching files
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Organize files into a dictionary
    extracted_files = {key: {'data': None, 'header': None} for key in target_strings}

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)

        for key in target_strings:
            if key in csv_file:
                if 'header' in csv_file:
                    extracted_files[key]['header'] = pd.read_csv(file_path, sep=';')
                else:
                    extracted_files[key]['data'] = pd.read_csv(file_path, sep=';')

    # Ensure all required files exist before processing
    if all(extracted_files[key]['data'] is not None and extracted_files[key]['header'] is not None for key in target_strings):
        folder_name = os.path.basename(directory)
        folder_data[folder_name] = {
            'XiY': extracted_files['xiy_fft_abs_avg'],
            'shift_fft': extracted_files['shift_val_fft_abs_avg'],
            'XiY_filter': extracted_files['xiy_fft_abs_filter']
        }

print(f"Total folders processed: {len(folder_data)}")

# %%

# Process data: Remove timestamp column and reshape
for folder_name, data_dict in folder_data.items():
    for key in ['XiY', 'shift_fft', 'XiY_filter']:  
        # Remove 'timestamp' column if it exists
        if 'timestamp' in data_dict[key]['data'].columns:
            data_dict[key]['data'].drop(columns=['timestamp'], inplace=True)

        # Convert DataFrame to NumPy array and assign back to 'data'
        if isinstance(data_dict[key]['data'], pd.DataFrame):
            data_dict[key]['data'] = data_dict[key]['data'].to_numpy()

        # Reshape the second column (values) based on unique values in the first column (index)
        first_column = data_dict[key]['data'][:, 0]  # First column (indices)
        second_column = data_dict[key]['data'][:, 1]  # Second column (values)

        # Get the number of unique values in the first column
        unique_indices = np.unique(first_column)

        # Reshape the second column into a 2D array with rows equal to the number of unique values
        reshaped_values = np.array([second_column[first_column == idx] for idx in unique_indices])

        # Replace the original second column with the reshaped 2D array
        data_dict[key]['data'] = np.mean(reshaped_values, axis=0)

# Compute the ratio and store it in a new key
for folder_name, data_dict in folder_data.items():
    xiy_data = data_dict['XiY']['data']
    filter_data = data_dict['XiY_filter']['data']

    # Ensure element-wise division is valid
    if xiy_data.shape == filter_data.shape:
        data_dict['ratio'] = {'data': xiy_data / filter_data}
    else:
        print(f"Shape mismatch in folder {folder_name}: {xiy_data.shape} vs {filter_data.shape}")


# %%

test_data = folder_data[folder_name]['shift_fft']['header']


#%%
XiY_grid_delta = folder_data[folder_name]['XiY']['header']['grid_col_delta'][0]
XiY_grid_columns = folder_data[folder_name]['XiY']['header']['grid_columns'][0]
XiY_grid_offset = folder_data[folder_name]['XiY']['header']['grid_col_offset'][0]
XiY_freq_domain = np.arange(XiY_grid_offset,XiY_grid_delta*XiY_grid_columns//2, XiY_grid_delta)
#%%

shift_grid_delta = folder_data[folder_name]['shift_fft']['header']['grid_col_delta'][0]
shift_grid_columns = folder_data[folder_name]['shift_fft']['header']['grid_columns'][0]
shift_grid_offset = folder_data[folder_name]['shift_fft']['header']['grid_col_offset'][0]
shift_freq_domain = np.arange(shift_grid_offset,shift_grid_delta*shift_grid_columns, shift_grid_delta)

# %%
import matplotlib.pyplot as plt

# Create a figure and axis for subplots
num_folders = len(folder_data)
fig, axes = plt.subplots(num_folders, 1, figsize=(10, 5 * num_folders))

# Ensure axes is always iterable, even if there's only one subplot
if num_folders == 1:
    axes = [axes]

# Dictionary to store results
results = {}

# Iterate through folder_data and plot the 'XiY' and 'shift_fft' data
for idx, (folder_name, data_dict) in enumerate(folder_data.items()):
    # Extract the data for 'XiY' and 'shift_fft' (both should be 1D arrays of shape (1024,))
    xiy_data_filt = data_dict['ratio']['data']
    shift_fft_data = data_dict['shift_fft']['data']

    # Compute middle index and extract the middle 100 points
    xiy_mid_index = len(XiY_freq_domain) // 2
    x_slice = XiY_freq_domain[xiy_mid_index - 50: xiy_mid_index + 50]
    xiy_slice = xiy_data_filt[xiy_mid_index - 50: xiy_mid_index + 50] #USE FILTER-COMPED DATA

    x_slice_fft = shift_freq_domain[0:50]
    shift_fft_slice = shift_fft_data[0:50]

    # **Find indices where x values are between 5 and 15**
    xiy_indices = (x_slice >= 5) & (x_slice <= 15)
    shift_fft_indices = (x_slice_fft >= 5) & (x_slice_fft <= 15)

    # Compute the mean values in the selected range
    avg_xiy = np.mean(xiy_slice[xiy_indices]) if np.any(xiy_indices) else None
    avg_shift_fft = np.mean(shift_fft_slice[shift_fft_indices]) if np.any(shift_fft_indices) else None

    # Compute Signal-to-Noise Ratio (SNR) for XiY
    max_xiy = np.max(xiy_slice)
    snr_xiy = max_xiy / avg_xiy if avg_xiy and avg_xiy > 0 else None

    # Store results
    results[folder_name] = {
        "avg_xiy": avg_xiy,
        "avg_shift_fft": avg_shift_fft,
        "snr_xiy": snr_xiy
    }

    # Create twin axes for the current subplot
    ax1 = axes[idx]
    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis

    # Plot 'XiY' data on the first y-axis (ax1)
    ax1.plot(x_slice, xiy_slice, label='XiY', color='blue')
    ax1.set_yscale('log')  # Set the first y-axis to log scale
    ax1.set_ylabel('XiY (Log scale)')  # Label for the first y-axis

    # Plot 'Shift FFT' data on the second y-axis (ax2)
    ax2.plot(x_slice_fft, shift_fft_slice, label='Shift FFT', color='red')
    ax2.set_yscale('log')  # Set the second y-axis to log scale
    ax2.set_ylabel('Shift FFT (Log scale)')  # Label for the second y-axis

    # **Plot the average values as horizontal lines**
    if avg_xiy is not None:
        ax1.axhline(avg_xiy, color='blue', linestyle='dashed', alpha=0.7, label=f'XiY Avg: {avg_xiy:.3e}')
    if avg_shift_fft is not None:
        ax2.axhline(avg_shift_fft, color='red', linestyle='dashed', alpha=0.7, label=f'Shift FFT Avg: {avg_shift_fft:.3e}')

    # Set common title and x-axis label
    ax1.set_title(f"Data for {folder_name}")
    ax1.set_xlabel('Frequency Domain')

    # Add legends for both y-axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

# Load data
loaded_data = np.load(r"W:\Data\2025_03_06_grad_power\Grad_power2.npz", allow_pickle=True)
results_loaded = dict(loaded_data)

# Extract values
power_values = results_loaded['power_values']
width_values = results_loaded['widths']
phase_values = results_loaded['phase_slopes']

# Extract data for plotting
folder_names = list(results.keys())  # Get folder names
snr_values = [results[folder]['snr_xiy'] for folder in folder_names]  # SNR values
avg_shift_values = [results[folder]['avg_shift_fft'] for folder in folder_names]  # Avg shift FFT values

# Create figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# ---- Subplot 1: SNR and Avg Shift FFT ----
ax1 = axes[0]
ax1.plot(power_values, snr_values, label='SNR (XiY)', color='blue', marker='o', linestyle='-')
ax1.set_ylabel('SNR (XiY)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Second y-axis for average shift FFT
ax2 = ax1.twinx()
ax2.plot(power_values, np.array(avg_shift_values)*0.071488e-9/1e-15*2, label='Average noise', color='red', marker='s', linestyle='--')
ax2.set_ylabel('Average noise level (fT/rHz)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.set_title('SNR and Average Noise vs. Power Delivered')

# ---- Subplot 2: Widths and Phase Slopes ----
ax3 = axes[1]
ax3.plot(power_values, width_values, label='Widths', color='green', marker='^', linestyle='-')

ax4 = ax3.twinx()
ax4.plot(power_values, phase_values, label='Phase Slopes', color='purple', marker='x', linestyle='--')

ax3.set_ylabel('Widths (Hz)', color='green')
ax3.tick_params(axis='y', labelcolor='green')
ax4.set_ylabel('Phase Slopes (degs/Hz)', color='purple')
ax4.tick_params(axis='y', labelcolor='purple')

ax3.set_xlabel('Power Delivered to Cells (uW)')
ax3.set_title('Widths and Phase Slopes vs. Power Delivered')

# Adjust layout and show
plt.tight_layout()
plt.show()

# %%

