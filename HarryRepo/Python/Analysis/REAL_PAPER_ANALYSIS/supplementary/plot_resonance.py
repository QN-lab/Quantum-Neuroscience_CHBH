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
# from Proc import obs
import obs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_16_bench\\splitting_res_1_000\\'     
# Folderpath_r  = 'Y:/OPM_data/whole_sweep/C1/'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_16_bench\\splitting_res_000\\'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_16_bench\\splitting_res_1_000\\'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_01_25\\session_20230125_235834_03\\finding_best_overlap_000\\'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_25_bench\\grad_100nT_bb_noise_res\\grad_bb_noise_res_000\\'
# Folderpath_r = 'Z:\\Data\\2023_08_17_bench\\grad_noise_res\\grad_noise_res_000\\'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_25_bench\\mag_100nT_bb_noise_res\\mag_bb_noise_res_000\\'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\grad_noise_res\\grad_noise_res_000\\'
# Folderpath_r = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\mag_noise_res\\mag_noise_res_000\\'

Filename_r = 'dev3994_demods_0_sample_00000.csv'
Headername_r = 'dev3994_demods_0_sample_header_00000.csv'

res, res_legends = obs.resReadData(Folderpath_r,Filename_r,Headername_r,';')

resonance = obs.Resonance(res,res_legends,1100)

resonance.plot_no_fit()

print(resonance.width)