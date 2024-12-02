import mne
import matplotlib.pyplot as plt
import numpy as np

# Base directory for data
base_directory = r'Y:\Harry_TMS\Projector_noise\sub-00000'

# Define filenames and common bad channels
filenames = [
    '20241118_171304_sub-00000_file-ProjOFF_raw.fif',
    '20241118_170926_sub-00000_file-ProjON_raw.fif',
    '20241118_173759_sub-00000_file-ProjONCovered2_raw.fif'
]
# bad_channels = ['s3_bz']  # Replace with actual bad channels

# Colors and labels for each run
colors = ['red', 'blue', 'green']
labels = ['OFF', 'ON', 'ON_covered']

# Define the list of frequencies where vertical lines will be plotted
line_noise = [50 * factor for factor in [1/3, 2/3, 1, 4/3, 5/3, 2]]

# Prepare storage for PSD data and time series data
psds = []
freqs_list = []
timecourses = []

# Load data and compute PSD for each file
for filename in filenames:
    raw = mne.io.read_raw_fif(base_directory + '\\' + filename, preload=True)
    # Set bad channels
    # raw.info['bads'] = bad_channels
    
    # Compute PSD
    psd = raw.compute_psd(fmin=0.1, fmax=150, tmin=0, tmax=None, picks='meg', n_fft=25000)
    freqs = psd.freqs
    psd_data = psd.get_data()  # PSD data for all channels in linear scale (Tesla/√Hz)
    psd_data = np.sqrt(psd_data)  # Square-root the PSD data
    psds.append(psd_data)
    freqs_list.append(freqs)
    
    # Extract time series data
    times = raw.times
    data, _ = raw[:, :]  # Extract all channel data
    timecourses.append((times, data))

# Create a figure with 3 subplots for PSD
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=True)

# Loop through each PSD data and subplot
for idx, (psd_data, freqs, color, label) in enumerate(zip(psds, freqs_list, colors, labels)):
    ax = axs[idx]  # Select subplot based on index

    # Plot each channel's PSD in the respective subplot
    for psd_line in psd_data:
        ax.plot(freqs, psd_line, color=color, alpha=0.5)  # Adjust alpha for transparency

    # Add vertical lines at the frequencies specified in line_noise
    for noise_freq in line_noise:
        ax.axvline(x=noise_freq, color='black', linestyle='--', linewidth=1, alpha=0.7)  # Adjust alpha for opacity

    # Set x-axis limits and y-axis limits
    ax.set_xlim(0, 150)
    ax.set_ylim(10e-15, 100e-12)  # Apply y-axis limits

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (T/√Hz)')  # Updated unit label
    ax.set_title(labels[idx])
    ax.set_yscale('log')
    ax.grid(True)

# Adjust layout and show the figure
plt.tight_layout()
plt.show()

# Plotting all runs in a single plot (combined plot)
plt.figure(figsize=(12, 8))

# Create a handle for the legend
handles = []

# Loop through each PSD data and plot combined
for psd_data, freqs, color, label in zip(psds, freqs_list, colors, labels):
    for i, psd_line in enumerate(psd_data):
        line, = plt.plot(freqs, psd_line, color=color, alpha=0.5)  # Adjust alpha for transparency
        if i == 0:  # Add a single handle for each run (only for the first channel)
            handles.append(line)

# Add legend only once per run
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (T/√Hz)')  # Updated unit label
plt.title('Noise Spectral Density for Different Runs')
plt.yscale('log')
plt.grid(True)
plt.xlim(0, 150)  # Ensure x-axis limits for the combined plot
plt.ylim(10e-15, 100e-12)  # Apply y-axis limits for combined plot
plt.show()

# Create a figure with a single plot for timecourse
plt.figure(figsize=(12, 8))

# Prepare for global y-axis limits
all_signals = []
for times, data in timecourses:
    avg_signal = np.mean(data, axis=0)
    # Baseline correction (mean of the first 10 seconds as baseline)
    baseline = np.mean(avg_signal)
    avg_signal -= baseline
    all_signals.append(avg_signal)

# Compute global y-axis limits
global_ylim = (1.1 * np.min([np.min(signal) for signal in all_signals]), 
                1.1 * np.max([np.max(signal) for signal in all_signals]))

# Plot all runs in a single plot
for (times, data, color, label) in zip(timecourses, [data for _, data in timecourses], colors, labels):
    avg_signal = np.mean(data, axis=0)
    baseline = np.mean(avg_signal)
    avg_signal -= baseline
    plt.plot(times[0], avg_signal, color=color, label=label)

plt.xlabel('Time (s)')
plt.ylabel('Baseline Corrected Signal Amplitude')
plt.title('Average Signal Across Channels for Different Runs')
plt.legend()
plt.grid(True)
plt.ylim(global_ylim)  # Apply global y-axis limits
plt.tight_layout()
plt.show()

#%% Fun

# Create a figure with a single plot for timecourse without axes and labels
# plt.figure(figsize=(12, 8),dpi=500)
# for (times, data, color, label) in zip(timecourses, [data for _, data in timecourses], colors, labels):
#     avg_signal = np.mean(data, axis=0)
#     baseline = np.mean(avg_signal)
#     avg_signal -= baseline
#     plt.plot(times[0], avg_signal, color=color, linewidth=17.5)  # Thicken lines

# # Set a white background and remove axes, labels, and legend
# plt.gca().set_facecolor('white')
# plt.axis('off')

# plt.tight_layout()
# plt.show()