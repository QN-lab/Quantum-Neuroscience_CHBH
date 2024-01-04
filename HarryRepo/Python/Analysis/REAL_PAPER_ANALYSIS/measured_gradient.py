# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:42:06 2023

@author: H
"""
from Proc import obs
# import obs
import matplotlib.pyplot as plt
import numpy as np
import regex as re
import pandas as pd
import os
import math
# import Harry_analysis as HCA
from scipy.fft import fft, fftfreq
plt.rcParams['text.usetex'] = True
plt.style.use('classic')

###############################################################################
#%% 
Gain = 14.2
T1=0
T2=3

base_directory_g = 'Z:\\jenseno-opm\\Data\\2023_08_18_bench\\grad_noise_high\\'
# base_directory_g = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\grad_noise\\'
# base_directory_g = 'Z:\\Data\\2023_08_25_bench\\grad_100nT_bb_noise\\'
subfolder_list_g = os.listdir(base_directory_g)

base_directory_m = 'Z:\\jenseno-opm\\Data\\2023_08_18_bench\\mag_noise_high\\'
# base_directory_m = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\mag_noise\\'
# base_directory_m = 'Z:\\Data\\2023_08_25_bench\\mag_100nT_bb_noise\\'
subfolder_list_m = os.listdir(base_directory_m)
 


print('Loading.....')

Data_list_g = list()
for cur_subfolder in subfolder_list_g:
    print('...')
    Data_list_g.append(obs.Joined(base_directory_g, cur_subfolder, Gain, T1, T2))
    print('Loaded ' + cur_subfolder)
    
    
Data_list_m = list()
for cur_subfolder in subfolder_list_m:
    print('...')
    Data_list_m.append(obs.Joined(base_directory_m, cur_subfolder, Gain, T1, T2))
    print('Loaded ' + cur_subfolder)


#%%

mV = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65])
# mV = np.array([0,3,6,9,12,15,18,21,24,27,30,33])
# mV = np.array([3,6,9,12,15,18,21,24,27,30,33])

b_field = mV/6.45

g_val = np.zeros(len(b_field))
for i in range(len(Data_list_g)):
    g_val[i] = Data_list_g[i].PiD.yf_avg_a[6]

#%% Extract 6Hz measured value

fig,ax = plt.subplots()
ax.plot(b_field,g_val/1e-12,'bo')
ax.set_xlabel('applied noise deltaB')
ax.set_ylabel('measured gradient (pT)')