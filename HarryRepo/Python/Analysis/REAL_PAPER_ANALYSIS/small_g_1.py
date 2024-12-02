# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:42:06 2023

@author: H
"""
from Proc import obs
# import obs
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
import regex as re
import pandas as pd
import os
import math
import Harry_analysis as HCA
from scipy.fft import fft, fftfreq

###############################################################################
#%%

Gain = 14.2
T1=0
T2=4

base_directory_g = 'Z:\\Data\\2023_08_11_bench\\low_g_grad\\'

subfolder_list_g = os.listdir(base_directory_g)
print('Loading.....')

Data_list_g = list()
for cur_subfolder in subfolder_list_g:
    print('...')
    Data_list_g.append(obs.Joined(base_directory_g, cur_subfolder, Gain, T1, T2))
    print('Loaded ' + cur_subfolder)

#%% 

Gsupp = 4*(2/3.07)

Tup = [0.05,0.95]
Tdown = [1.05,1.95]
inx_up = np.searchsorted(Data_list_g[0].PiD.chunked_time[0,:], Tup,side='left')
inx_down = np.searchsorted(Data_list_g[0].PiD.chunked_time[0,:], Tdown,side='left')

up_dat = Data_list_g[0].PiD.RoI[:,inx_up[0]:inx_up[1]]
down_dat = Data_list_g[0].PiD.RoI[:,inx_down[0]:inx_down[1]]

run_av_up = np.mean(up_dat,axis=1)
run_std_up = np.std(up_dat,axis=1)

run_av_down = np.mean(down_dat,axis=1)
run_std_down = np.std(down_dat,axis=1)

diffs = run_av_down-run_av_up

av_change = np.mean(diffs)
std_dev = np.std(diffs)

print('Measured Change:'+str(av_change/1e-12)+' pT std.dev: '+str(std_dev/1e-12)+' pT')

for i in range(Data_list_g[1].PiD.Field.shape[0]):
    plt.figure()
    plt.plot(Data_list_g[0].PiD.chunked_time[0,:],Data_list_g[1].PiD.Field[i,:])

plt.figure()
plt.stackplot(range(up_dat.shape[1]),up_dat)