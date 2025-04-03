import mne
import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import array
from datetime import datetime

#%% Preprop

#Load up
directory_path = 'Y:/Harry_TMS/Auditory'    #Folder location
files = os.listdir(directory_path)          #List present files
file_name= '20241211_163259_sub-Pilot2_file-Auditory1_raw.fif'
data_path=os.path.join(directory_path,file_name)

# Check if file exists for redundancy
file_exists = os.path.isfile(data_path)

#Read and extract events
raw = mne.io.read_raw_fif(data_path, preload=True)
events = mne.find_events(raw, stim_channel='di32')

#Determine events: 2bit input is the oddball trigger, 4bit is the normal tone
events_ids = {"oddball": 2, "tone":4}

raw.pick('meg')
projs = mne.preprocessing.compute_proj_hfc(raw.info, order=2,exclude=['s16_bz']) #exclude more bad sensors here.
raw.add_proj(projs).apply_proj(verbose="error")

raw.plot(scalings={"mag":  10e-9}) #Scaling here is wrong I think, not super important

#Look at events to see if there's enough and that they make sense.
fig = mne.viz.plot_events(
    events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_ids
)

#%%Epoching

#I think timings are off for now..

#epoch variables
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

#The sensor names have a bunch of useless numbers and letters in front of them so I'll remove them

    #Get the original channel names
ch_names = epochs.info['ch_names']

# Create a dictionary to rename channels by removing the first 8 characters to make more sense
rename_dict = {name: name[8:] for name in ch_names}

    #Rename the channels
epochs.rename_channels(rename_dict)

    #topomap sensor names in locations, so that I can see them.
epochs.plot_sensors(kind="3d", ch_type="all")
epochs.plot_sensors(kind="topomap", ch_type="all",show_names=True)

#ANY SENSORS AT LOCATION 0,0,0 failed to localise, remove them if needed.

#Simple PSD
psd = epochs.compute_psd(
    fmax=60, 
    method="multitaper",  # Multitaper is the default
    bandwidth=1           # Smaller value for higher resolution
)
psd.plot(picks="meg", exclude="bads", amplitude=True)


#Plot evoked resononses on a topo to see where we get the most signal (should be around auditory cortex)
evoked= epochs.copy().average(method='mean').filter(0.0, 30).crop(-0.1,0.8)

evoked.plot(time_unit="s")

evoked.copy().apply_baseline(baseline=(-0.1, 0))
evoked.copy().pick('mag').plot_topo()