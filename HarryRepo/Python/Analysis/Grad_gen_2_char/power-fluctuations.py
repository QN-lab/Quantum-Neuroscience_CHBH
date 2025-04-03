# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:25:52 2025

@author: vpixx
"""

import os
import pandas as pd
import csv
import matplotlib.pyplot as plt

directory = r'C:\Users\vpixx\Documents\Thorlabs\Optical Power Monitor'
all_files = os.listdir(directory)


csv_files = [file for file in all_files if file.endswith('.csv')]
filenames = [f for f in os.listdir(directory) if f.endswith('.csv')]


dataframes = []
for csv_file in csv_files:
    file_path = os.path.join(directory, csv_file)
    try:
        with open(file_path, 'r') as f:
            # Detect delimiter
            dialect = csv.Sniffer().sniff(f.read(1024))
            f.seek(0)
            # Load CSV with the detected delimiter
            df = pd.read_csv(file_path, delimiter=dialect.delimiter, skip_blank_lines=True)
        dataframes.append(df)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

import numpy as np

# Initialize an empty list to collect the 'Power (W)' values
power_values = []

power_values_list = [
    df['Power (W)'].to_numpy() if 'Power (W)' in df.columns else np.array([])
    for df in dataframes
]

# Determine the maximum number of values (m)
max_length = max(len(values) for values in power_values_list)

seconds = np.arange(0,max_length,1)
# Create a 2D array filled with NaN, with shape (n, m)
power_array = np.full((len(dataframes), max_length), np.nan)

# Fill the array with the values from each DataFrame
for i, values in enumerate(power_values_list):
    power_array[i, :len(values)] = values


# Create a new plot for percent change normalized data
plt.figure(figsize=(10, 6))

# Loop through rows in 'power_array'
for i in range(power_array.shape[0]):
    # Calculate the percent change from the initial value
    percent_change = ((power_array[i] - power_array[i][0]) / power_array[i][0]) * 100
    
    # Plot the percent change
    plt.plot(seconds, percent_change, label=f'Percent Change Line {i+1}')

# Add labels and legend
plt.xlabel('Seconds')
plt.ylabel('Percent Change in Power (%)')
plt.title('Percent Change in Power vs Time')
plt.legend(filenames)
plt.grid(True)

# Show the plot
plt.show()