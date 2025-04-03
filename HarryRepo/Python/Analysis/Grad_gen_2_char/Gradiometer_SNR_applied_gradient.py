
#%%

import os
import pandas as pd
import numpy as np

# Directory containing the files
core_path = r'W:\Data\2025_03_10_gradiometer_measure_gradient\spect'

# List all directories inside core_path
all_directories = [
    os.path.join(core_path, folder)
    for folder in os.listdir(core_path)
    if os.path.isdir(os.path.join(core_path, folder))
]
# Sort first by string length, then alphabetically within the same length
all_directories.sort(key=lambda x: (len(os.path.basename(x)), x))

# Target strings to match
target_strings = ['xiy_fft_abs_avg', 'shift_val_fft_abs_avg', 'xiy_fft_abs_filter','auxin0_fft_abs_avg']

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
            'XiY_filter': extracted_files['xiy_fft_abs_filter'],
            'PD_fft': extracted_files['auxin0_fft_abs_avg']
        }

print(f"Total folders processed: {len(folder_data)}")

# %%

# Process data: Remove timestamp column and reshape
for folder_name, data_dict in folder_data.items():
    for key in ['XiY', 'shift_fft', 'XiY_filter','PD_fft']:  
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

        # If key is 'PD_fft', add the '0Hz_amplitude' entry
        if key == 'PD_fft':
            data_dict[key]['0Hz_amplitude'] = np.max(data_dict[key]['data'])

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

    # **Find indices where x values are between 10 and 25**
    xiy_indices = (x_slice >= 10) & (x_slice <= 25)
    shift_fft_indices = (x_slice_fft >= 10) & (x_slice_fft <= 25)

    # Compute the mean values in the selected range
    avg_xiy = np.mean(xiy_slice[xiy_indices]) if np.any(xiy_indices) else None
    avg_shift_fft = np.mean(shift_fft_slice[shift_fft_indices]) if np.any(shift_fft_indices) else None

    # Compute Signal-to-Noise Ratio (SNR) for XiY
    max_xiy = np.max(xiy_slice)
    snr_xiy = max_xiy / avg_xiy if avg_xiy and avg_xiy > 0 else None

 # **Find the SNR for shift_fft at 7Hz**
    shift_fft_7Hz_index = np.argmin(np.abs(x_slice_fft - 7))  # Find closest index to 7Hz
    signal_at_7Hz = shift_fft_slice[shift_fft_7Hz_index]  # Extract corresponding signal
    snr_shift_fft = signal_at_7Hz / avg_shift_fft if avg_shift_fft and avg_shift_fft > 0 else None

    # Store results
    results[folder_name] = {
        "avg_xiy": avg_xiy,
        "avg_shift_fft": avg_shift_fft,
        "snr_xiy": snr_xiy,
        "snr_shift_fft": snr_shift_fft  # Add new SNR value for shift_fft
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

    # **Plot a vertical line at 7Hz for verification**
    ax1.axvline(x=7, color='green', linestyle='dotted', alpha=0.8, label='7Hz Marker')

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

# Extract data for plotting
pd_amplitudes = []
snr_shift_fft_values = []
snr_xiy_values = []
shift_fft_noise_values = []

for folder_name, data_dict in folder_data.items():
    if 'PD_fft' in data_dict and '0Hz_amplitude' in data_dict['PD_fft']:
        pd_amplitudes.append(data_dict['PD_fft']['0Hz_amplitude'])
        snr_shift_fft_values.append(results[folder_name]['snr_shift_fft'])
        snr_xiy_values.append(results[folder_name]['snr_xiy'])
        shift_fft_noise_values.append(results[folder_name]['avg_shift_fft'])  # Assuming noise is stored as avg_shift_fft

# Convert to numpy arrays for easier plotting
pd_amplitudes = np.exp((np.array(pd_amplitudes) * 1000 - 209.3080) / 41.31)  # Convert to linear units
snr_shift_fft_values = np.array(snr_shift_fft_values)
snr_xiy_values = np.array(snr_xiy_values)
shift_fft_noise_values = (np.array(shift_fft_noise_values) **2) *0.071e-9 *14.7# Convert to uHz/rHz

# Find max PLL SNR index and corresponding x-value
max_snr_index = np.argmax(snr_shift_fft_values)
max_snr_x = pd_amplitudes[max_snr_index]

# Create a new figure with two subplots sharing the same x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Plot XiY SNR on the first subplot
ax1.plot(pd_amplitudes, snr_xiy_values, 'bo-', label="SNR XiY")
ax1.set_ylabel("SNR XiY", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title("XiY SNR vs Power Delivered to Cells")
ax1.grid(True)
#ax1.legend()

# Plot PID SNR and Shift FFT Noise on the second subplot
ax2.plot(pd_amplitudes, snr_shift_fft_values, 'ro-')
ax2.set_ylabel("PLL SNR @ 7Hz", color='red')
ax2.tick_params(axis='y', labelcolor='red')
#ax2.legend(loc='upper left')

ax3 = ax2.twinx()  # Create second y-axis for Shift FFT Noise
ax3.plot(pd_amplitudes, shift_fft_noise_values, 'go--', label="Shift FFT Noise (urHz)")
ax3.set_ylabel("PLL Noise (urHz)", color='green')
ax3.tick_params(axis='y', labelcolor='green')
#ax3.legend(loc='upper right')

# Add vertical line at max PLL SNR
ax2.axvline(max_snr_x, color='black', linestyle='dashed', label=f'Max SNR @ {max_snr_x:.2f} uW')
ax2.legend()

# Add vertical line at max PLL SNR
ax1.axvline(max_snr_x, color='black', linestyle='dashed', label=f'Max SNR @ {max_snr_x:.2f} uW')

ax2.set_xlabel("Assumed power before cells (uW)")
ax2.set_title("PID SNR and PID Noise vs Power Delivered to Cells")
ax2.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# %%
