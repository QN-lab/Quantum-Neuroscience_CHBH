#%% Load in
import mne
import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import array
from datetime import datetime

directory_path = 'Y:/Harry_TMS/Motor'
files = os.listdir(directory_path)
file_name= '20241211_161104_sub-Pilot2_file-Motor1_raw.fif'
data_path=os.path.join(directory_path,file_name)

# Check if the specific file exists
file_exists = os.path.isfile(data_path)

raw = mne.io.read_raw_fif(data_path, preload=True)
events = mne.find_events(raw, stim_channel='di32')
event_id = {
    "misc": 2,
    "misc2": 4,
    "stim_on": 64,
    "stim_off": 208
}

raw.plot(events=events,start=5,duration=10,scalings={"mag":  1e-9})

raw.pick('meg')
projs = mne.preprocessing.compute_proj_hfc(raw.info, order=2,exclude=['s16_bz'])
raw.add_proj(projs).apply_proj(verbose="error")

#%% Epochs

tmin, tmax = -0.4, 2
lowpass, highpass = 40, 0.1
baseline_tmin, baseline_tmax = -0.4, -0.2  # None takes the first timepoint

raw_filtered = raw.copy().filter(highpass, lowpass)

epochs = mne.Epochs(
    raw_filtered,
    events,
    event_id,
    tmin=tmin,
    tmax=tmax,
    detrend=1,
    reject=dict(mag=5e-11),
    baseline=(baseline_tmin,baseline_tmax),
    preload=True,
)


print(epochs.events)  # Displays the events used for epoching
print(epochs.event_id)  # Confirms which event ID was used

# Get the original channel names
ch_names = epochs.info['ch_names']

# Create a dictionary to rename channels by removing the first 8 characters
rename_dict = {name: name[8:] for name in ch_names}

# Rename the channels
epochs.rename_channels(rename_dict)

epochs.plot_sensors(kind="3d", ch_type="all")
epochs.plot_sensors(kind="topomap", ch_type="all",show_names=True)

psd = epochs["stim_on"].compute_psd(
    fmax=60, 
    method="multitaper",  # Multitaper is the default
    bandwidth=1          # Smaller value for higher resolution
)
psd.plot(picks="meg", exclude="bads", amplitude=True)

freqs = np.arange(10, 31, 1)
n_cycles = freqs / 2 
time_bandwidth = 2.0

tfr_stim_on =  mne.time_frequency.tfr_multitaper(
    epochs['stim_on'], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'mag',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=2,
    n_jobs = -1,
    verbose=True)

tfr_stim_on.plot(
    picks=['s75'], 
    tmin=-0.5, tmax=1.2, 
    title='s81')  


plt = tfr_stim_on.plot_topo(
    tmin=-0.3, tmax=1.2, 
    baseline=[-0.3,-0.2], 
    mode="percent", 
    fig_facecolor='w',
    font_color='k',
    vmin=-1, vmax=1,
    title='TFR of power <30 Hz')