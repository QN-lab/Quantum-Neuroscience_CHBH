# -*- coding: utf-8 -*-
"""
@author: H
"""

from Proc import obs
import Harry_analysis as HCA
import os
import mne
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True
plt.style.use('default')

pd.get_option("display.max_columns",15)

###################################################################################################
base_directory= 'Z:\\Data\\2023_04_19_Zurich_Brain\\'

#%% Plot Resonance
# res_dir = base_directory+'resonance\\grad_1_res_000\\'

# r_sig = 'dev3994_demods_0_sample_00000.csv'
# r_header = 'dev3994_demods_0_sample_header_00000.csv'


# res_g, res_legends_g = HCA.ReadData(res_dir,r_sig,r_header,';')
# resonance_g = obs.Resonance(res_g, res_legends_g, 1100)

# res_f = resonance_g.data[0,:,0]
# res_x = resonance_g.data[0,:,1]/1e-3
# res_y = resonance_g.data[0,:,2]/1e-3
# res_phi = resonance_g.data[0,:,3]

# fig,ax= plt.subplots()

# ax.plot(res_f,res_x,c='green')
# ax.plot(res_f,res_y,c='blue')
# ax.axhline(y=0,c='k',ls='--')
# ax.grid()

# ax.set(xlabel='Modulation Frequency (Hz)',ylabel='Rotation (arb,mV)',
#        ylim=[min(res_x)-1,max(res_x)+1],xlim=[min(res_f)-100,max(res_f)+100])


#%% Plot Runs

def ReadData(cur_fold,sigs,headers):
    #Read in signals and headers
    out_sig = list()
    out_headers = list()
    for i in range(len(sigs)):
        out_sig.append(pd.read_csv(cur_fold+sigs[i],sep=';'))
        out_headers.append(pd.read_csv(cur_fold+headers[i],sep=';'))
    return out_headers, out_sig
 

headers = ['dev3994_demods_0_sample_x_avg_header_00000.csv',
             'dev3994_demods_0_sample_y_avg_header_00000.csv',
             'dev3994_pids_0_stream_shift_avg_header_00000.csv'
             ]

sigs = ['dev3994_demods_0_sample_x_avg_00000.csv',              #X
          'dev3994_demods_0_sample_y_avg_00000.csv',            #Y
          'dev3994_pids_0_stream_shift_avg_00000.csv'           #PID Signal
              ]

gain = 14.2

#CHANGE THIS ONE
run_dir = r'Z:\Data\2023_08_17_bench\DC_steps\mag_dc_2.0-2.5nT_6Hz_5pTcm_000'

extract = ReadData(run_dir+'\\',sigs,headers)


demodX = extract[1][0]['value']/1e-3
demodY = extract[1][1]['value']/1e-3 
pll = extract[1][1]['value']*gain*0.071488e-9 #Check


# Assuming all arrays have the same length
length = len(demodX)  # or len(demodY), or len(pll)

# Sampling frequency
fs = 837.1  # Hz

# Calculate time array
time = np.arange(length) / fs  # Create an array from 0 to (length-1)/fs

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot demodX
axs[0].plot(time, demodX, label='demodX')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('demodX Value')
axs[0].legend()
axs[0].grid(True)

# Plot demodY
axs[1].plot(time, demodY, label='demodY', color='orange')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('demodY Value')
axs[1].legend()
axs[1].grid(True)

# Plot pll
axs[2].plot(time, pll, label='pll', color='green')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('B-Field')
axs[2].legend()
axs[2].grid(True)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

'''
potential runs to look at:
    
Z:\Data\2023_04_28_Brain_after_fixing\runs\grad_1_000
Z:\Data\2023_04_28_Brain_after_fixing\runs\grad_2_000
Z:\Data\2023_04_28_Brain_after_fixing\runs\grad_empty_25BW_000
Z:\Data\2023_08_17_bench\DC_steps\grad_dc_0.5-1.0nT_6Hz_5pTcm_000
Z:\Data\2023_08_17_bench\DC_steps\grad_dc_1.25-1.75nT_6Hz_5pTcm_000
Z:\Data\2023_08_17_bench\DC_steps\grad_dc_2.0-2.5nT_6Hz_5pTcm_000 <---------------- clipping

'''


