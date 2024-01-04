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

#%% LOAD FIELDLINE
folder_fl = 'Z:/jenseno-opm/Data/2023_03_01/1/20230301/3/'
fname_fl = '20230301_183051_3_1_lifts_1_raw.fif'

raw = mne.io.read_raw_fif(folder_fl+fname_fl)
info = mne.io.read_info(folder_fl+fname_fl)
print(info)

string = ['9,Lift Diagonal','34, X','57, Y', '19, CHBH Diagonal','93, Z']

data, times = raw.get_data(return_times=True)

times = times
data = data[1:]

for i in range(0,5):
    plt.figure()
    plt.plot(times,data[i,:])
    plt.title(string[i])
    plt.ylabel('Field (nT)')
    plt.xlabel('Time (s)')
plt.show()
