#%% Import and plot

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Use Qt5Agg to plot outside VS Code terminal
matplotlib.use("Qt5Agg")

core_path = 'W:\\Data\\2025_01_30_bias_vs_solenoid\\'

# Filter directories that start with "C2", contain "_freq", and do NOT contain "_mV"
all_directories = [
    os.path.join(core_path, folder)
    for folder in os.listdir(core_path)
    if os.path.isdir(os.path.join(core_path, folder)) and 
       folder.startswith('C2') and 
       '_freq' in folder and 
       '_mV' not in folder
]

folder_data_pairs = []

# Loop to collect data from header and sample files
for directory in all_directories:
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    header_file = None
    sample_file = None

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)

        if 'header' in csv_file:
            header_file = pd.read_csv(file_path, sep=';')
        else:
            sample_file = pd.read_csv(file_path, sep=';')

    if header_file is not None and sample_file is not None:
        folder_name = os.path.basename(directory)
        folder_data_pairs.append((header_file, sample_file, folder_name))

print(f"Total folders processed: {len(folder_data_pairs)}")

# Initialize dictionaries to store data for each folder
x_data_dict = {}
y_data_dict = {}
phase_data_dict = {}
freq_data_dict = {}

# Loop through each folder, extract all instances, and compute mean
for header_file, sample_file, folder_name in folder_data_pairs:
    freq_inds = sample_file.index[sample_file['fieldname'] == 'frequency'].tolist()
    x_inds = sample_file.index[sample_file['fieldname'] == 'x'].tolist()
    y_inds = sample_file.index[sample_file['fieldname'] == 'y'].tolist()
    phase_inds = sample_file.index[sample_file['fieldname'] == 'phase'].tolist()  # Extract phase_ data

    if not freq_inds or not x_inds or not y_inds or not phase_inds:
        print(f"Skipping {folder_name} due to missing 'f', 'x', 'y', or 'phase' data.")
        continue

    # Extract all rows for frequency, x, y, and phase
    freq_data = sample_file.iloc[freq_inds, 4:-1].to_numpy()
    x_data = sample_file.iloc[x_inds, 4:-1].to_numpy()
    y_data = sample_file.iloc[y_inds, 4:-1].to_numpy()
    phase_data = sample_file.iloc[phase_inds, 4:-1].to_numpy()

    # Compute mean and standard deviation for x, y, and phase
    freq_mean = np.mean(freq_data, axis=0)
    x_mean = np.mean(x_data, axis=0)
    x_std = np.std(x_data, axis=0)
    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0)
    phase_mean = np.mean(phase_data, axis=0)
    phase_std = np.std(phase_data, axis=0)

    # Store mean and std for x, y, and phase
    freq_data_dict[folder_name] = freq_mean
    x_data_dict[folder_name] = (x_mean, x_std)
    y_data_dict[folder_name] = (y_mean, y_std)
    phase_data_dict[folder_name] = (phase_mean, phase_std)  # Store phase data

print("Data extraction and averaging complete!")

# Get the number of folders processed
num_folders = len(freq_data_dict)

# Determine subplot grid size (make it as square as possible)
rows = int(np.ceil(np.sqrt(num_folders)))  # Number of rows
cols = int(np.ceil(num_folders / rows))  # Number of columns

# Create the figure and subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True)
axes = np.array(axes).flatten()  # Flatten to ensure it's a 1D list

# Loop through each folder and plot the mean with shaded std for x, y, and phase
for i, (folder_name, freq_mean) in enumerate(freq_data_dict.items()):
    x_mean, x_std = x_data_dict[folder_name]
    y_mean, y_std = y_data_dict[folder_name]
    phase_mean, phase_std = phase_data_dict[folder_name]  # Get phase mean and std

    ax = axes[i]  # Get the subplot for this folder

    # Plot mean with shaded ±1 standard deviation for x (in orange)
    ax.plot(freq_mean, x_mean, label="X Mean (Amps)", color='orange')
    ax.fill_between(freq_mean, x_mean - x_std, x_mean + x_std, color='orange', alpha=0.2)

    # Plot mean with shaded ±1 standard deviation for y (in blue)
    ax.plot(freq_mean, y_mean, label="Y Mean (Amps)", color='blue', linestyle='dashed')
    ax.fill_between(freq_mean, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.2)

    # Create a second y-axis for phase
    ax2 = ax.twinx()  # Create twin axis for phase

    # Plot phase mean with shaded ±1 standard deviation (in red)
    ax2.plot(freq_mean, phase_mean, label="Phase Mean (radians)", color='red', linestyle='dotted')
    ax2.fill_between(freq_mean, phase_mean - phase_std, phase_mean + phase_std, color='red', alpha=0.2)

    # Set labels and legends
    ax.set_title(folder_name, fontsize=10)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("X, Y values (Amps)")
    ax.legend(loc='upper left')
    
    ax2.set_ylabel("Phase (radians)", color='red')
    ax2.legend(loc='upper right')

    ax.grid(True)

# Remove any empty subplots
for i in range(num_folders, len(axes)):
    fig.delaxes(axes[i])  # Remove unused subplot spaces

# Adjust layout
plt.tight_layout()
plt.suptitle("Mean X, Y, and Phase as a Function of Frequency (±1 Std Dev)", fontsize=14, y=1.02)
plt.show()

#%% Lorentzian fits and plotting
from scipy.optimize import curve_fit
# Define Lorentzian function
def Lorentzian(x, amp, cen, wid, slope, offset):
    return (amp * (wid)**2) / ((x - cen)**2 + (wid)**2) + slope * x + offset

# Store the amplitude-to-width ratios and fitted parameters for each folder
ratio_dict = {}
fitted_params_dict = {}

# Create a figure for the overlayed Lorentzian fits
plt.figure(figsize=(10, 6))

# Loop through each folder and apply the Lorentzian fit to the averaged x data
for _, _, folder_name in folder_data_pairs:
    # Access the folder's frequency and x data
    freq_mean = freq_data_dict[folder_name]
    x_mean, x_std = x_data_dict[folder_name]

    # Initial guess for the Lorentzian fit parameters: [amp, cen, wid, slope, offset]
    initial_guess = [max(x_mean), 3000, 100, 0, min(x_mean)]

    # Fit the Lorentzian function to the averaged data
    popt, _ = curve_fit(Lorentzian, freq_mean, x_mean, p0=initial_guess)

    # Extract the fitted parameters: amp, cen, wid, slope, offset
    amp, cen, wid, slope, offset = popt

    # Convert HWHM to FWHM
    wid = wid * 2

    # Convert amplitude from amps to nano-amps (multiply by 1e9)
    amp_nano = amp * 1e9  # nano-amps

    # Compute the amplitude-to-width ratio
    amp_to_width_ratio = amp_nano / wid

    # Store the ratio and fitted parameters for future use
    ratio_dict[folder_name] = amp_to_width_ratio
    fitted_params_dict[folder_name] = (amp_nano, cen, wid, slope, offset)

    # Modify the folder_name in the legend to include only "bias-coil" or "solenoid" if present
    legend_name = ''
    if 'bias-coil' in folder_name:
        legend_name = 'bias-coil'
    elif 'solenoid' in folder_name:
        legend_name = 'solenoid'

    # Generate the fitted Lorentzian curve using the optimized parameters
    fitted_x = Lorentzian(freq_mean, *popt)

    # Plot the fitted Lorentzian curve for this folder
    plt.plot(freq_mean, fitted_x, label=f'{legend_name} (Amp={amp_nano:.2f} nA, Width={wid:.2f}, Ratio={amp_to_width_ratio:.2f})')

# Add plot labels and title for the Lorentzian fits
plt.xlabel("Frequency (Hz)")
plt.ylabel("X-fit (Amps)")
plt.title("Overlayed Lorentzian Fits")
plt.legend()

# Display the plot for Lorentzian fits
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Phase plotting within cen ± 2*wid (raw data) and fitting within cen ± wid/4

# Define a linear fitting function for phase
def linear_fit(x, slope, offset):
    return slope * x + offset

# Create a new plot for phase data within cen ± 2*wid
plt.figure(figsize=(10, 6))

# Different line styles for the dotted lines
line_styles = ['k--', 'b:', 'g-.', 'r-.', 'm--']

# Loop through each folder and plot the phase_mean data within the defined frequency range
for idx, (_, _, folder_name) in enumerate(folder_data_pairs):
    # Access the folder's frequency and phase data
    freq_mean = freq_data_dict[folder_name]
    phase_mean, _ = phase_data_dict[folder_name]  # Get phase mean (std is not needed here)

    # Retrieve the fitted parameters from the previous section
    amp_nano, cen, wid, slope, offset = fitted_params_dict[folder_name]  # Retrieve from fitted_params_dict
   
    # Define the frequency range for the raw phase plot (cen ± 2*wid)
    lower_bound_raw = cen - (2 * wid)
    upper_bound_raw = cen + (2 * wid)

    # Find the indices of the frequencies that fall within the specified range (raw data)
    indices_in_range_raw = np.where((freq_mean >= lower_bound_raw) & (freq_mean <= upper_bound_raw))[0]

    # Extract the frequency and phase data within the specified range (raw data)
    freq_range_raw = freq_mean[indices_in_range_raw]
    phase_range_raw = phase_mean[indices_in_range_raw]

    legend_name = ''
    if 'bias-coil' in folder_name:
        legend_name = 'bias-coil'
    elif 'solenoid' in folder_name:
        legend_name = 'solenoid'

    # Plot the 'raw' phase data within the specified range (cen ± 2*wid)
    plt.plot(freq_range_raw, phase_range_raw, label=f'{legend_name} Raw Phase')

    # Define the frequency range for fitting (cen ± wid/4)
    lower_bound_fit = cen - (wid / 4)
    upper_bound_fit = cen + (wid / 4)

    # Find the indices of the frequencies that fall within the fitting range (smaller range)
    indices_in_range_fit = np.where((freq_mean >= lower_bound_fit) & (freq_mean <= upper_bound_fit))[0]

    # Extract the frequency and phase data within the fitting range
    freq_range_fit = freq_mean[indices_in_range_fit]
    phase_range_fit = phase_mean[indices_in_range_fit]

    # Fit the phase data within this range
    popt, _ = curve_fit(linear_fit, freq_range_fit, phase_range_fit)
    slope_fit, offset_fit = popt

    # Generate the fitted phase curve
    fitted_phase = linear_fit(freq_range_fit, slope_fit, offset_fit)

    # Use different line styles for each fit
    line_style = line_styles[idx % len(line_styles)]

    # Plot the fitted phase curve as a dotted line with distinct style
    plt.plot(freq_range_fit, fitted_phase, line_style, label=f'{legend_name} Fit (Slope={slope_fit:.3f})')

# Add plot labels and title for the phase data
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.title("Phase within Cen ± 2*Wid and Fitted Phase within Cen ± Wid/4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

