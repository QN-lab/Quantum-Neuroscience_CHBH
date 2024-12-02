# -*- coding: utf-8 -*-
"""
@author: H
"""
from Proc import obs
# import obs
import matplotlib
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import regex as re
import pandas as pd
import os
import math
# import Harry_analysis as HCA
from scipy.fft import fft, fftfreq
plt.rcParams['text.usetex'] = True
plt.style.use('default')

###############################################################################

#%% 
# Gain = 14.2
Gain = 14.2*2 #DOUBLED! CAREFULLY. Seems like if we measure a gradient, we don't want double, but for noise we do. 

T1=0
T2=3

# base_directory_g = 'Z:\\Data\\2023_08_18_bench\\grad_noise_high\\'
base_directory_g = 'Z:\\Data\\2023_08_17_bench\\grad_noise\\'
# base_directory_g = 'Z:\\Data\\2023_08_25_bench\\grad_100nT_bb_noise\\'
subfolder_list_g = os.listdir(base_directory_g)

# base_directory_m = 'Z:\\Data\\2023_08_18_bench\\mag_noise_high\\'
base_directory_m = 'Z:\\Data\\2023_08_17_bench\\mag_noise\\'
# base_directory_m = 'Z:\\Data\\2023_08_25_bench\\mag_100nT_bb_noise\\'
subfolder_list_m = os.listdir(base_directory_m)

print('Loading.....')

Data_list_g = list()
for cur_subfolder in subfolder_list_g:
    print('...')
    Data_list_g.append(obs.Joined(base_directory_g, cur_subfolder, Gain, T1, T2))
    print('Loaded ' + cur_subfolder)
    
    
Data_list_m = list()
for cur_subfolder in subfolder_list_m:
    print('...')
    Data_list_m.append(obs.Joined(base_directory_m, cur_subfolder, Gain, T1, T2))
    print('Loaded ' + cur_subfolder)


#%%

# mV = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65]) #grad_100nT_bb_noise
mV = np.array([0,3,6,9,12,15,18,21,24,27,30,33]) #grad_noise
# mV = np.array([3,6,9,12,15,18,21,24,27,30,33]) #grad_noise_high

b_field = mV/6.45

###################################################################################################
#Gradiometer first
grad_freqs_l = list()

for i , field_g in enumerate(b_field):
    grad_freqs = Data_list_g[i].PiD.yf_chunked_a
    grad_freqs_l.append(grad_freqs)

# Create a 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Parameters for the stacking
z_spacing = 10  # Spacing between heatmaps in the z-direction

zlabel = 'Noise Amplitude (nT)'

# Desired frequency range
x_min = 1
x_max = 60

# We need to store the min and max values for the color normalization
min_value = np.inf
max_value = -np.inf

# # First loop to find the min and max values for normalization
# for grad_freqs in grad_freqs_l:
#     min_value = min(min_value, np.min(grad_freqs))
#     max_value = max(max_value, np.max(grad_freqs))
min_value = 1e-15
max_value = 1e-10

# Create the mappable object with the colormap and normalization
norm = LogNorm(vmin=min_value, vmax=max_value)
mappable = ScalarMappable(norm=norm, cmap=plt.cm.viridis)

# Loop over the heatmaps and plot each one at a different z-level
num_heatmaps = len(grad_freqs_l)

# Second loop to plot the surfaces
for i, grad_freqs in enumerate(grad_freqs_l):
    N, M = grad_freqs.shape  # Get the shape of the current heatmap
    
    # Define the x indices corresponding to the desired frequency range
    x_indices = np.arange(x_min, min(M, x_max))  # Ensure not to exceed array bounds
    
    # Slice the data to only include the desired frequency range
    sliced_grad_freqs = grad_freqs[:, x_indices]
    
    # Define the x and y grids for the sliced data
    x = np.arange(len(x_indices))  # Number of frequency points after slicing
    y = np.arange(N)  # Number of spectra
    x, y = np.meshgrid(x, y)
    
    # Reverse the z level (height) to flip the plot
    z = np.full_like(x, (num_heatmaps - i - 1) * z_spacing)
    
    # Plot the surface with the sliced heatmap data at the defined z level
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.viridis(norm(sliced_grad_freqs)), shade=False)

# Customize the view angle and labels
ax.view_init(elev=30, azim=-60)  # Adjust view angle
ax.set_xlabel('Frequency')
ax.set_ylabel('Run number')
ax.set_zlabel(zlabel)

# Set the x-axis limits to match the sliced data
ax.set_xlim([0, len(x_indices)])

z_ticks = np.flip((np.arange(len(b_field)) * z_spacing))
ax.set_zticks(z_ticks)

# Set custom z-tick labels
ax.set_zticklabels([f'{val:.1f}' for val in b_field])

# Add the colorbar to the figure
mappable.set_array([])  # This line is needed to allow colorbar creation without actual data
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)  # Adjust `shrink` and `aspect` for size
cbar.set_label('Signal Strength(T)')  # Label for the colorbar

plt.show()

###################################################################################################
#Magnetometer

mag_freqs_l = list()

for i , field_m in enumerate(b_field):
    mag_freqs = Data_list_m[i].PiD.yf_chunked_a
    mag_freqs_l.append(mag_freqs)

# Create a 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Parameters for the stacking
z_spacing = 10  # Spacing between heatmaps in the z-direction

# Desired frequency range
x_min = 1
x_max = 60

# Loop over the heatmaps and plot each one at a different z-level
num_heatmaps = len(grad_freqs_l)

# We need to store the min and max values for the color normalization
min_value = np.inf
max_value = -np.inf

# # First loop to find the min and max values for normalization
# for mag_freqs in mag_freqs_l:
#     min_value = min(min_value, np.min(mag_freqs))
#     max_value = max(max_value, np.max(mag_freqs))

min_value = 1e-15
max_value = 1e-10

# Create the mappable object with the colormap and normalization
norm = LogNorm(vmin=min_value, vmax=max_value)
mappable = ScalarMappable(norm=norm, cmap=plt.cm.viridis)

# Loop over the heatmaps and plot each one at a different z-level
num_heatmaps = len(mag_freqs_l)

for i, mag_freqs in enumerate(mag_freqs_l):
    N, M = mag_freqs.shape  # Get the shape of the current heatmap
    
    # Define the x indices corresponding to the desired frequency range
    x_indices = np.arange(x_min, min(M, x_max))  # Ensure not to exceed array bounds
    
    # Slice the data to only include the desired frequency range
    sliced_mag_freqs = mag_freqs[:, x_indices]
    
    # Define the x and y grids for the sliced data
    x = np.arange(len(x_indices))  # Number of frequency points after slicing
    y = np.arange(N)  # Number of spectra
    x, y = np.meshgrid(x, y)
    
    # Reverse the z level (height) to flip the plot
    z = np.full_like(x, (num_heatmaps - i - 1) * z_spacing)
    
    # Plot the surface with the sliced heatmap data at the defined z level
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.viridis(LogNorm()(sliced_mag_freqs)), shade=False)
    
# Customize the view angle and labels
ax.view_init(elev=30, azim=-60)  # Adjust view angle
ax.set_xlabel('Frequency')
ax.set_ylabel('Run number')
ax.set_zlabel(zlabel)

# Set the x-axis limits to match the sliced data
ax.set_xlim([0, len(x_indices)])

z_ticks = np.flip((np.arange(len(b_field))*z_spacing))
ax.set_zticks(z_ticks)

# Set custom z-tick labels
ax.set_zticklabels([f'{val:.1f}' for val in b_field])

# Add the colorbar to the figure
mappable.set_array([])  # This line is needed to allow colorbar creation without actual data
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)  # Adjust `shrink` and `aspect` for size
cbar.set_label('Signal Strength(T)')  # Label for the colorbar

plt.show()

#%% Trim runs so they contain the same number of trials

assert len(mag_freqs_l)==len(grad_freqs_l)

#combine list to equalise the number of rows
combined_list = [grad_freqs_l + mag_freqs_l][0]

#fixing to the minimum number of trials per run
num_rows = [array.shape[0] for array in combined_list]

min_rows = min(num_rows)


def trim_array(array, min_rows):
    return array[:min_rows,:]

#Trim every array within the combined list to only contain the max number of shared columns
trimmed_list = [trim_array(array, min_rows) for array in combined_list]

#carefully re-extracting grad from mag in the combined list
g_trim = trimmed_list[:len(grad_freqs_l)]

m_trim = trimmed_list[len(grad_freqs_l):]

#Subtract g from m

diff_trim = [20*np.log10(arr1/arr2) for arr1, arr2 in zip(m_trim, g_trim)]

# diff_trim = [20*np.log10()]

# Create a 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Parameters for the stacking
z_spacing = 10  # Spacing between heatmaps in the z-direction

# Desired frequency range
x_min = 1
x_max = 60

# Loop over the heatmaps and plot each one at a different z-level
num_heatmaps = len(diff_trim)

for i, diff_freqs in enumerate(diff_trim):
    N, M = diff_freqs.shape  # Get the shape of the current heatmap
    
    # Define the x indices corresponding to the desired frequency range
    x_indices = np.arange(x_min, min(M, x_max))  # Ensure not to exceed array bounds
    
    # Slice the data to only include the desired frequency range
    sliced_diff_freqs = diff_freqs[:, x_indices]
    
    # Define the x and y grids for the sliced data
    x = np.arange(len(x_indices))  # Number of frequency points after slicing
    y = np.arange(N)  # Number of spectra
    x, y = np.meshgrid(x, y)
    
    # Reverse the z level (height) to flip the plot
    z = np.full_like(x, (num_heatmaps - i - 1) * z_spacing)
    
    # Normalize the data for linear colormap
    norm = Normalize(vmin=np.min(sliced_diff_freqs), vmax=np.max(sliced_diff_freqs))
    
    # Plot the surface with the sliced heatmap data at the defined z level
    surf = ax.plot_surface(
        x, y, z, 
        rstride=1, cstride=1, 
        facecolors=plt.cm.viridis(norm(sliced_diff_freqs)), 
        shade=False,
        linewidth=0
    )
    
# Customize the view angle and labels
ax.view_init(elev=30, azim=-60)  # Adjust view angle
ax.set_xlabel('Frequency')
ax.set_ylabel('Run number')

# Set the x-axis limits to match the sliced data
ax.set_xlim([0, len(x_indices)])
ax.set_zlabel(zlabel)

z_ticks = np.flip((np.arange(len(b_field))*z_spacing))
ax.set_zticks(z_ticks)

# Set custom z-tick labels
ax.set_zticklabels([f'{val:.2f}' for val in b_field])

# Add a colorbar for linear colormap
mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
mappable.set_array([])
fig.colorbar(mappable, ax=ax, label='Attenuation (dB)')

plt.show()

#%%Extracting SNR from point-by-point then averaging

#Array-ift [b_field,run#,freq_data]

g_trim_arr = np.stack(g_trim)
m_trim_arr = np.stack(m_trim)

sig_g = g_trim_arr[:, :, 6] 
sig_m = m_trim_arr[:, :, 6]

noise_all_g = g_trim_arr[:, :, 7:48]
noise_all_m = m_trim_arr[:, :, 7:48]

# Averaging over spectrum within each run
noise_avg_g = np.mean(noise_all_g, axis=2)
noise_avg_m = np.mean(noise_all_m, axis=2)

snr_all_g = sig_g / noise_avg_g
snr_all_m = sig_m / noise_avg_m

# Create the plot

fig, axs = plt.subplots(figsize=(12, 6))  # Width is increased to 12 inches, height kept at 6

# Plot the SNR data with larger markers
axs.plot(b_field, snr_all_m, 'ro', alpha=0.05, markersize=15)  # Increase markersize for the red points
axs.plot(b_field, snr_all_g, 'bo', alpha=0.05, markersize=15)  # Increase markersize for the blue points

# Plot the mean SNR for each sensor type with larger markers
axs.plot(b_field, np.mean(snr_all_m, axis=1), 'r+', label='Magnetometer', markersize=20)  # Larger mean SNR markers
axs.plot(b_field, np.mean(snr_all_g, axis=1), 'b+', label='Gradiometer', markersize=20)  # Larger mean SNR markers

# Add a horizontal line at y=1 to mark unity SNR
axs.axhline(y=1, color='green', label='unity SNR (unreliable)')

# Set axis labels, limits, and font sizes
axs.set_xlabel(zlabel, fontsize=18)  # Adjust font size for x-axis label
axs.set_ylabel('SNR', fontsize=18)   # Adjust font size for y-axis label
axs.set_ylim([0, 100])
axs.set_xlim([0.1, 5.5])

# Add legend with increased font size
axs.legend(fontsize=18)  # Set legend font size to match the axes labels

# Add grid
axs.grid()

# Show the plot
plt.show()




#%% Average

atten_all = np.stack(diff_trim)

avg_across_trials = np.mean(atten_all, axis=1)

plt.figure(figsize=(10, 8))  # Adjust the figure size if needed

# Clip the data to enhance contrast
clipped_data = np.clip(avg_across_trials[1:], a_min=-10, a_max=30)  # Adjust these values as needed

vmin_value = 0  # Set the minimum value for the colormap
vmax_value = 32  # Set the maximum value for the colormap

# Display the data as a heatmap with custom vmin and vmax
plt.imshow(clipped_data, aspect='auto', cmap='plasma_r', origin='lower', vmin=vmin_value, vmax=vmax_value)

# Add colorbar to indicate the amplitude values
cbar = plt.colorbar(label='Attenuation (dB)')
cbar.ax.tick_params(labelsize=14)  # Increase the colorbar tick label font size
cbar.set_label('Attenuation (dB)', fontsize=16)  # Increase the colorbar label font size

# Label axes with increased font size
plt.xlabel('Frequency', fontsize=16)
plt.ylabel(zlabel, fontsize=16)
plt.title('Attenuation with increased Field', fontsize=18)

# Increase tick label font sizes
plt.xticks(fontsize=14)
plt.yticks(ticks=np.arange(len(b_field[1:])), labels=[f"{value:.2f}" for value in b_field[1:]], fontsize=14)

# Set x-axis limit
plt.xlim([0, 60])

# Show the plot
plt.show()

#%% Plotting a couple time-series to see effects.
sig_select_g = np.stack([obj.PiD.Field[0] for obj in Data_list_g])
time_select_g = np.stack([obj.PiD.chunked_time[0] for obj in Data_list_g])

sig_select_m = np.stack([obj.PiD.Field[0] for obj in Data_list_m])
time_select_m = np.stack([obj.PiD.chunked_time[0] for obj in Data_list_m])

sample_end = 2000  # Select up to this sample

# Select the time range for plotting
single_time = time_select_g[0, :sample_end]

# Perform baseline correction by subtracting the mean of each run for both datasets
baseline_corrected_sig_g = sig_select_g[:, :sample_end] - sig_select_g[:, :sample_end].mean(axis=1, keepdims=True)
baseline_corrected_sig_m = sig_select_m[:, :sample_end] - sig_select_m[:, :sample_end].mean(axis=1, keepdims=True)

baseline_corrected_sig_g /= 1e-12
baseline_corrected_sig_m /= 1e-12

# Determine the number of runs to plot
number_of_runs = 3  

# Create subplots with 2 columns
fig, axs = plt.subplots(number_of_runs, 2, figsize=(15, 2 * number_of_runs), sharex=True)

# Initialize lists to store the max values for each column
hex_colour = '#9E58AF'  # Example: a bright orange colour

# Set the desired font sizes for the subplot titles and axis tick labels
title_fontsize = 20
tick_label_fontsize = 16

# Initialize lists to store the max values for each column
max_values_g = []
max_values_m = []

# Plot each run in its own subplot in the first column (sig_select_g)
for i in range(number_of_runs):
    axs[i, 0].plot(single_time, baseline_corrected_sig_g[i, :], color=hex_colour)
    axs[i, 0].set_title(f'Applied Noise: {b_field[i]:.2f}nT', fontsize=title_fontsize)
    axs[i, 0].grid(True)  # Optional: add a grid for better readability
    
    # Adjust the font size for the axis tick labels
    axs[i, 0].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    
    # Store the max value for this run
    max_values_g.append(np.max(baseline_corrected_sig_g[i, :]))

# Plot each run in its own subplot in the second column (sig_select_m)
for i in range(number_of_runs):
    axs[i, 1].plot(single_time, baseline_corrected_sig_m[i, :], color=hex_colour)
    axs[i, 1].set_title(f'Applied Noise: {b_field[i]:.2f}nT', fontsize=title_fontsize)
    axs[i, 1].grid(True)  # Optional: add a grid for better readability
    
    # Adjust the font size for the axis tick labels
    axs[i, 1].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    
    # Store the max value for this run
    max_values_m.append(np.max(baseline_corrected_sig_m[i, :]))

# Set y-axis limits based on the maximum value
max_limit_g = max(max_values_g)
max_limit_m = max(max_values_m)

for ax in axs[:, 0]:  # For the first column
    ax.set_ylim([-max_limit_g, max_limit_g])

for ax in axs[:, 1]:  # For the second column
    ax.set_ylim([-max_limit_m, max_limit_m])

# Move the y-axis label further to the left
fig.text(0.02, 0.5, 'Signal (pT)', va='center', rotation='vertical', fontsize=24)

# Move the x-axis label further down
fig.text(0.45, -0.02, 'Time (s)', va='center', fontsize=24)

# Add super-titles for each column
fig.text(0.29, 1.05, 'Gradiometer', ha='center', va='center', fontsize=24, fontweight='bold')
fig.text(0.725, 1.05, 'Magnetometer', ha='center', va='center', fontsize=24, fontweight='bold')

# Adjust the layout to make space for the super-titles
plt.tight_layout(rect=[0.05, 0, 0.95, 1.02])

# Show the plot
plt.show()




