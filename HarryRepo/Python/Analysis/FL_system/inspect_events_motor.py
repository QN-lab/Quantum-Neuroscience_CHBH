import os
import numpy as np
import mne

directory_path = 'Y:/Harry_TMS/Motor'
files = os.listdir(directory_path)
file_name= '20241211_161104_sub-Pilot2_file-Motor1_raw.fif'
data_path=os.path.join(directory_path,file_name)

raw = mne.io.read_raw_fif(data_path, verbose=False)
raw.crop(tmin=50,tmax=60).load_data()
events = mne.find_events(raw, stim_channel='di32')

fig = mne.viz.plot_events(
    events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)