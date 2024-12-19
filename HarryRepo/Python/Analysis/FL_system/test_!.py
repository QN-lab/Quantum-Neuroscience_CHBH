import mne
import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import array
from datetime import datetime

directory_path = 'Y:/Harry_TMS/Auditory'
files = os.listdir(directory_path)
file_name= '20241211_163259_sub-Pilot2_file-Auditory1_raw.fif'
data_path=os.path.join(directory_path,file_name)

# Check if the specific file exists
file_exists = os.path.isfile(data_path)

raw = mne.io.read_raw_fif(data_path, preload=True)
events = mne.find_events(raw, stim_channel='di32')

events_ids = {"oddball": 2, "tone":4}

raw.pick('meg')

raw.plot(scalings={"mag":  10e-9})

tmin, tmax = -0.4, 0.8
lowpass, highpass = 40, 0.1
baseline_tmin, baseline_tmax = -0.4, -0.2  # None takes the first timepoint

raw_filtered = raw.copy().filter(highpass, lowpass)

epochs = mne.Epochs(
    raw_filtered,
    events,
    events_ids,
    tmin=tmin,
    tmax=tmax,
    detrend=1,
    reject=dict(mag=5e-11),
    baseline=(baseline_tmin,baseline_tmax),
    preload=True,
)

print(epochs)

# Get the original channel names
ch_names = epochs.info['ch_names']

# Create a dictionary to rename channels by removing the first 8 characters
rename_dict = {name: name[8:] for name in ch_names}

# Rename the channels using the rename_channels method
epochs.rename_channels(rename_dict)

epochs.plot_sensors(kind="3d", ch_type="all")
epochs.plot_sensors(kind="topomap", ch_type="all",show_names=True)

psd = epochs.compute_psd(
    fmax=60, 
    method="multitaper",  # Multitaper is the default
    bandwidth=1          # Smaller value for higher resolution
)
psd.plot(picks="meg", exclude="bads", amplitude=True)


evoked= epochs.copy().average(method='mean').filter(0.0, 30).crop(-0.1,0.8)

evoked.plot(time_unit="s")

evoked.copy().apply_baseline(baseline=(-0.1, 0))
evoked.copy().pick('mag').plot_topo();