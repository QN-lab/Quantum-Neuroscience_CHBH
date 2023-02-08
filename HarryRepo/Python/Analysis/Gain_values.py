# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:51:51 2023

@author: vpixx
"""

import matplotlib.pyplot as plt#
import numpy as np

#%% Magnetometer
conv = 0.071488 #nT/Hz
amp = 0.5 # nT
exp_freq_pp = (2*amp)/conv

Filter = np.array([10,25,50,100,150,200])
pp_freq = np.array([0.165,0.160,0.155,0.145,0.140,0.135])
freq_err = 0.005

gain_ratio = exp_freq_pp/pp_freq
gain_err = gain_ratio*(freq_err/pp_freq)

plt.figure()
plt.errorbar(Filter,pp_freq,yerr=freq_err,fmt='b*')
plt.xlim(0,250)
plt.ylim(0.125,0.175)
plt.xlabel('Filter Bandwidth (Hz)')
plt.ylabel('Measured P-P Frequency (Hz)')
plt.title('Filter BW vs measured field in Hz')
plt.grid(color='k', linestyle='-', linewidth=0.3)

plt.figure()
plt.errorbar(Filter,gain_ratio,yerr=gain_err,fmt='b*')
plt.xlim(0,250)
plt.ylim(80,110)
plt.xlabel('Filter Bandwidth (Hz)')
plt.ylabel('Gain (arb)')
plt.title('Gain required to meet expected field measurement at different filters')
plt.grid(color='k', linestyle='-', linewidth=0.3)

#%%Gradiometer
conv = 71.488 #pT/Hz
amp = 75 # pT/cm
exp_freq_pp = (2*amp)/conv
Filter = np.array([10,25,50,100,150,200])
pp_freq = np.array([0.375,0.372,0.355,0.350,0.345,0.330])
freq_err = 0.01

gain_ratio = exp_freq_pp/pp_freq
gain_err = gain_ratio*(freq_err/pp_freq)

plt.figure()
plt.errorbar(Filter,pp_freq,yerr=freq_err,fmt='b*')
plt.xlim(0,250)
# plt.ylim(0.125,0.175)
plt.xlabel('Filter Bandwidth (Hz)')
plt.ylabel('Measured P-P Frequency (Hz)')
plt.title('Filter BW vs measured field in Hz')
plt.grid(color='k', linestyle='-', linewidth=0.3)

plt.figure()
plt.errorbar(Filter,gain_ratio,yerr=gain_err,fmt='b*')
plt.xlim(0,250)
# plt.ylim(80,110)
plt.xlabel('Filter Bandwidth (Hz)')
plt.ylabel('Gain (arb)')
plt.title('Gain required to meet expected field measurement at different filters')
plt.grid(color='k', linestyle='-', linewidth=0.3)