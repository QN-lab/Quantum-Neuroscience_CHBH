import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#%%

# Define file paths
c1_laser_path = r"W:\Data\2025_02_26_laser_power\Cell1_laser.npz"  
c2_laser_path = r"W:\Data\2025_02_28_Cell2_laser_power\Cell2_laser.npz"  

# Load data
c1_laser = np.load(c1_laser_path)
c2_laser = np.load(c2_laser_path)

print("Data loaded successfully from:", c1_laser_path)
print("Data loaded successfully from:", c2_laser_path)

# Extract variables
c1_power_values = c1_laser["power_values"]
c1_widths = c1_laser["widths"]
c1_amplitudes = c1_laser["amplitudes"]
c1_amplitude_width_ratios = c1_laser["amplitude_width_ratios"]
c1_phase_slopes = c1_laser["phase_slopes"]

c2_power_values = c2_laser["power_values"]
c2_widths = c2_laser["widths"]
c2_amplitudes = c2_laser["amplitudes"]
c2_amplitude_width_ratios = c2_laser["amplitude_width_ratios"]
c2_phase_slopes = c2_laser["phase_slopes"]

#%% Figure 1: amps and widths
# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Plot width against power values
axes[0].plot(c1_power_values, c1_widths, marker='o', linestyle='-', color='b', label="Cell 1")
axes[0].plot(c2_power_values, c2_widths, marker='o', linestyle='-', color='r', label="Cell 2")
axes[0].set_ylabel("Width(Hz)")
axes[0].set_title("Width vs Power")
axes[0].grid(True)
axes[0].legend()

# Plot amplitude against power values
axes[1].plot(c1_power_values, c1_amplitudes, marker='s', linestyle='-', color='b', label="Cell 1")
axes[1].plot(c2_power_values, c2_amplitudes, marker='s', linestyle='-', color='r', label="Cell 2")
axes[1].set_ylabel('Amplitudes (nA)')
axes[1].set_title("Amplitude vs Power")
axes[1].grid(True)
axes[1].legend()

# Plot amplitude/width ratio against power values
axes[2].plot(c1_power_values, c1_amplitude_width_ratios, marker='^', linestyle='-', color='b', label="Cell 1")
axes[2].plot(c2_power_values, c2_amplitude_width_ratios, marker='^', linestyle='-', color='r', label="Cell 2")
axes[2].set_xlabel("Assumed power before first cell (uW)")
axes[2].set_ylabel("Amplitude/Width Ratio (nA/Hz)")
axes[2].set_title("Amplitude/Width Ratio vs Power")
axes[2].grid(True)
axes[2].legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# %% SECOND FIGURE
fig2, axes2 = plt.subplots(2, 1, figsize=(8, 12))

# Plot fitted phase slope against power values
axes2[0].plot(c1_power_values, c1_phase_slopes, marker='^', linestyle='-', color='b', label="Cell 1")
axes2[0].plot(c2_power_values, c2_phase_slopes, marker='^', linestyle='-', color='r', label="Cell 2")
axes2[0].set_xlabel("Assumed power before first cell (uW)")
axes2[0].set_ylabel("Absolute Phase Slope (degs/Hz)")
axes2[0].set_title("Phase Slope vs Power")
axes2[0].grid(True)
axes2[0].legend()

# Second subplot: Phase slope per width
axes2[1].plot(c1_power_values, np.array(c1_phase_slopes), marker='^', linestyle='-', color='b', label="Cell 1 Phase Slope")
axes2[1].plot(c2_power_values, np.array(c2_phase_slopes), marker='^', linestyle='-', color='r', label="Cell 2 Phase Slope")
axes2[1].set_xlabel("Assumed power before first cell (uW)")
axes2[1].set_ylabel("Phase slope (degs/Hz)")
axes2[1].set_title("Phase Slope Against Width")
axes2[1].grid(True)

# Create a second y-axis sharing the same x-axis
ax2 = axes2[1].twinx()
ax2.plot(c1_power_values, np.array(c1_widths)/2, marker='o', linestyle='--', color='c', label="Cell 1 Half-Widths")
ax2.plot(c2_power_values, np.array(c2_widths)/2, marker='o', linestyle='--', color='m', label="Cell 2 Half-Widths")
ax2.set_ylabel("Half-Width (Hz)") 
ax2.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# %%

# Define file paths
c1_heat_path = r"W:\Data\2025_02_27_heating\Cell1_heat.npz"  
c2_heat_path = r"W:\Data\2025_03_03_Cell2_heating\Cell2_heat.npz"  

# Load data
c1_heat = np.load(c1_heat_path)
c2_heat = np.load(c2_heat_path)

print("Data loaded successfully from:", c1_heat_path)
print("Data loaded successfully from:", c2_heat_path)

# Extract variables
c1_heat_values = c1_heat["power_values"]
c1_widths = c1_heat["widths"]
c1_amplitudes = c1_heat["amplitudes"]
c1_amplitude_width_ratios = c1_heat["amplitude_width_ratios"]
c1_phase_slopes = c1_heat["phase_slopes"]

c2_heat_values = c2_heat["power_values"]
c2_widths = c2_heat["widths"]
c2_amplitudes = c2_heat["amplitudes"]
c2_amplitude_width_ratios = c2_heat["amplitude_width_ratios"]
c2_phase_slopes = c2_heat["phase_slopes"]

#%% Figure 1: amps and widths
# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

# Plot width against heat values
axes[0].plot(c1_heat_values, c1_widths, marker='o', linestyle='-', color='b', label="Cell 1")
axes[0].plot(c2_heat_values, c2_widths, marker='o', linestyle='-', color='r', label="Cell 2")
axes[0].set_ylabel("Width(Hz)")
axes[0].set_title("Width vs heat")
axes[0].grid(True)
axes[0].legend()

# Plot amplitude against heat values
axes[1].plot(c1_heat_values, c1_amplitudes, marker='s', linestyle='-', color='b', label="Cell 1")
axes[1].plot(c2_heat_values, c2_amplitudes, marker='s', linestyle='-', color='r', label="Cell 2")
axes[1].set_ylabel('Amplitudes (nA)')
axes[1].set_title("Amplitude vs Heating")
axes[1].grid(True)
axes[1].legend()

# Plot amplitude/width ratio against heat values
axes[2].plot(c1_heat_values, c1_amplitude_width_ratios, marker='^', linestyle='-', color='b', label="Cell 1")
axes[2].plot(c2_heat_values, c2_amplitude_width_ratios, marker='^', linestyle='-', color='r', label="Cell 2")
axes[2].set_xlabel("Heating (C)")
axes[2].set_ylabel("Amplitude/Width Ratio (nA/Hz)")
axes[2].set_title("Amplitude/Width Ratio vs Heating")
axes[2].grid(True)
axes[2].legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# %% SECOND FIGURE
fig2, axes2 = plt.subplots(2, 1, figsize=(8, 12))

# Plot fitted phase slope against heat values
axes2[0].plot(c1_heat_values, c1_phase_slopes, marker='^', linestyle='-', color='b', label="Cell 1")
axes2[0].plot(c2_heat_values, c2_phase_slopes, marker='^', linestyle='-', color='r', label="Cell 2")
axes2[0].set_xlabel("Heating(C)")
axes2[0].set_ylabel("Absolute Phase Slope (degs/Hz)")
axes2[0].set_title("Phase Slope vs Power")
axes2[0].grid(True)
axes2[0].legend()

# Second subplot: Phase slope per width
axes2[1].plot(c1_heat_values, np.array(c1_phase_slopes), marker='^', linestyle='-', color='b', label="Cell 1 Phase Slope")
axes2[1].plot(c2_heat_values, np.array(c2_phase_slopes), marker='^', linestyle='-', color='r', label="Cell 2 Phase Slope")
axes2[1].set_xlabel("Heating(C)")
axes2[1].set_ylabel("Phase slope (degs/Hz)")
axes2[1].set_title("Phase Slope Against Width")
axes2[1].grid(True)

# Create a second y-axis sharing the same x-axis
ax2 = axes2[1].twinx()
ax2.plot(c1_heat_values, np.array(c1_widths)/2, marker='o', linestyle='--', color='c', label="Cell 1 Half-Widths")
ax2.plot(c2_heat_values, np.array(c2_widths)/2, marker='o', linestyle='--', color='m', label="Cell 2 Half-Widths")
ax2.set_ylabel("Half-Width (Hz)") 
ax2.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
# %%
