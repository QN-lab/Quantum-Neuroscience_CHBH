# -*- coding: utf-8 -*-
"""
@author: hxc214
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
mpl.use('Qt5Agg')
import math
# import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd
import mne
from os import listdir
from os.path import isfile, join

#%% LOAD FIELDLINE
sfreq = 1000

path = 'Z:\\jenseno-opm\\Data\\2023_03_28_FL\\'
files = [f for f in listdir(path) if isfile(join(path, f))]

# badpath = 'Z:\\jenseno-opm\\Data\\2023_03_27_FL\\syn_grad\\not_recorded\\'
# badfiles = [f for f in listdir(badpath) if isfile(join(badpath, f))]

# regex1 = '1_(.*)mVpp'
# regex2 = '1_(.*)mVpp_grad_3cm_away'

raws = list()
events = list()
epochs = list()
evoked = list()
data = list()

for i in range(len(files)):
    raws.append(mne.io.read_raw_fif(path+files[i]))
    events.append(mne.find_events(raws[i],stim_channel='Input1', min_duration= 0.001001))
    epochs.append(mne.Epochs(raws[i],
        events[i].astype(int), event_id=None,
        tmin=-0.20 , tmax=1,
        baseline=(-0.2,0),
        proj=True,
        picks = 'all',
        detrend = 1,
        reject_by_annotation=True,
        preload=True,
        verbose=True))
    evoked.append(epochs[i].average())
    data.append(evoked[i].get_data())
    
for i in range(len(files)):
    plt.figure()
    plt.plot(evoked[i].times,data[i][1]-(-1*data[i][0]),label = 'grad')
    plt.title(files[i])
    plt.xlabel('time (s)')
    plt.ylabel('Field (T)')
    plt.legend()
    




















