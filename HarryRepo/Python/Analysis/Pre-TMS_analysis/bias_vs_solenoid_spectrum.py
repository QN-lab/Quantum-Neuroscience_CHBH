
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")

core_path = 'W:\\Data\\2025_01_30_bias_vs_solenoid\\spect\\'

all_directories = [
    os.path.join(core_path, d) for d in os.listdir(core_path) 
    if os.path.isdir(os.path.join(core_path, d))
]

print(all_directories)

folder_data_pairs = []

# Loop to collect data from header and sample files
for directory in all_directories:
    csv_files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])  # Sort to maintain order

    header_files = [f for f in csv_files if 'header' in f]
    sample_files = [f for f in csv_files if 'header' not in f]

    if len(header_files) == len(sample_files):  # Ensure proper pairing
        folder_name = os.path.basename(directory)
        pairs = []

        for h_file, s_file in zip(header_files, sample_files):
            header_path = os.path.join(directory, h_file)
            sample_path = os.path.join(directory, s_file)

            header_df = pd.read_csv(header_path, sep=';')
            sample_df = pd.read_csv(sample_path, sep=';')

            pairs.append((header_df, sample_df, folder_name))

        folder_data_pairs.extend(pairs)  # Add all pairs from this folder

print(f"Total CSV pairs processed: {len(folder_data_pairs)}")

