# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:22:10 2023

@author: vpixx
"""

import numpy as np
import sys
#import sounddevice as sd
#import serial
#from simple_pid import PID
#import winsound
import regex as re
from Proc import obs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_16_bench\\splitting_res_1_000\\'     
# Folderpath_r  = 'Y:/OPM_data/whole_sweep/C1/'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_16_bench\\splitting_res_000\\'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_16_bench\\splitting_res_1_000\\'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_01_25\\session_20230125_235834_03\\finding_best_overlap_000\\'

Folderpath_r = 'Z:\\Data\\2023_08_17_bench\\mag_noise_res\\mag_noise_res_000\\'

Filename_r = 'dev3994_demods_0_sample_00000.csv'
Headername_r = 'dev3994_demods_0_sample_header_00000.csv'

res, res_legends = obs.PhaseReadData(Folderpath_r,Filename_r,Headername_r,';')

f_ind = res.index[res['fieldname'] == 'frequency'].tolist()
p_ind = res.index[res['fieldname'] == 'phase'].tolist()

f_data = res.iloc[f_ind[1],4:-1].to_numpy()
p_data = res.iloc[p_ind[1],4:-1].to_numpy()

dat = np.zeros((2,len(f_data)))

dat[0,:] = f_data
dat[1,:] = p_data

plt.plot(dat[0,:],dat[1,:])


def Sigmoid(x, a, b):
    return b*(((2)/(1 + np.exp(float(a) * x))) - 1)

param,cov = curve_fit(Sigmoid,f_data,p_data)

a = param[0]
b = param[1]

plt.plot(f_data,Sigmoid(dat[0,:],*param))
