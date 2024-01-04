# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:08:08 2023

@author: hxc214
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
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

path = 'Z:\\jenseno-opm\\Data\\2023_03_27_FL\\characterise_dipole\\'
files = [f for f in listdir(path) if isfile(join(path, f))]

ind_3 = [1,2,5,6,9,10]
ind_10 = [0,3,4,7,8,11]

# raw = mne.io.read_raw_fif(path+files[0])
# info = mne.io.read_info(path+files[0])
# print(info)
regex = 't_(.*)cm'

x = list()
for i in range(len(files)):
    x.append(re.findall(regex, files[i]))


raws = list()
events = list()
epochs = list()
evoked = list()
data = list()

for i in range(len(files)):
    raws.append(mne.io.read_raw_fif(path+files[i]))
    events.append(mne.find_events(raws[i],stim_channel='Input1',min_duration= 100/raws[i].info['sfreq']))
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
    
# for i in ind_3:
#     plt.figure()
#     plt.plot(evoked[i].times,data[i][0],label = 'Sensor1')
#     plt.plot(evoked[i].times,-1*data[i][1],label = 'Sensor2')
#     plt.title('{}cm Distance from Dipole(3mV applied)'.format(x[i][0]))
#     plt.xlabel('time (s)')
#     plt.ylabel('Field (T)')
#     plt.legend()
#     plt.legend()
    
for i in ind_10:
    plt.figure()
    plt.plot(evoked[i].times,data[i][0],label = 'Sensor1')
    plt.plot(evoked[i].times,-1*data[i][1],label = 'Sensor2')
    plt.title('{}cm from Dipole(10mV applied)'.format(x[i][0]))
    plt.xlabel('time (s)')
    plt.ylabel('Field (T)')
    plt.legend()

