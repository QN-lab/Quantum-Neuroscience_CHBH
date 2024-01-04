# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:42:06 2023

@author: H
"""
import obs
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
import regex as re
import pandas as pd
import os
import math
# import Harry_analysis as HCA
from scipy.fft import fft, fftfreq


#%% 

B = np.array([0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5])

Gain = 14.2
T1=0
T2=4

folder_directory ='Z:\\jenseno-opm\\Data\\2023_08_17_bench\\DC_steps\\'


headers = ['dev3994_demods_0_sample_auxin1_avg_header_00000.csv',
                 'dev3994_demods_0_sample_trigin2_avg_header_00000.csv',
                 'dev3994_demods_0_sample_xiy_fft_abs_avg_header_00000.csv',
                 'dev3994_pids_0_stream_shift_avg_header_00000.csv'
                 ]

sigs = ['dev3994_demods_0_sample_auxin1_avg_00000.csv',             #Arduino
              'dev3994_demods_0_sample_trigin2_avg_00000.csv',      #Trigger
              'dev3994_demods_0_sample_xiy_fft_abs_avg_00000.csv',  #XiY
              'dev3994_pids_0_stream_shift_avg_00000.csv'           #PID Signal
              ]

data_g = obs.Joined(folder_directory, 'grad_dc_2.0-2.5nT_6Hz_5pTcm_000',Gain,T1,T2)
data_m = obs.Joined(folder_directory, 'mag_dc_2.0-2.5nT_6Hz_5pTcm_000',Gain,T1,T2)

#%%
sr = 837.1

in_inx = round(0.4*sr)

time = data_g.PiD.chunked_time[0,:]
grad_data_i = data_g.PiD.Field
mag_data_i = data_m.PiD.Field

grad_bl = np.mean(grad_data_i[:,0:in_inx],axis=1)
mag_bl = np.mean(mag_data_i[:,0:in_inx],axis=1)

g_bl = np.array(grad_data_i.shape[1]*[grad_bl]).transpose()

m_bl = np.array(mag_data_i.shape[1]*[mag_bl]).transpose()

grad_data_bl = np.subtract(grad_data_i,g_bl)
mag_data_bl = np.subtract(mag_data_i,m_bl)

grad_data = np.mean(grad_data_bl,axis=0)
mag_data = np.mean(mag_data_bl,axis=0)



#%%

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_ylabel('Gradiometer',color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlabel('Time (s)')

# ax1.plot(time,grad_data,'b')
for i in range(grad_data_bl.shape[0]):
    ax1.plot(time,grad_data_bl[i],'b')

color = 'tab:green'

ax2 = plt.twinx(ax1)
ax2.set_ylabel('Magnetometer',color=color)
ax2.tick_params(axis='y', labelcolor=color)

# ax2.plot(time,-1*mag_data,color=color)
for i in range(mag_data_bl.shape[0]):
    ax2.plot(time,-1*mag_data_bl[i],color=color)
    
ylim = [-0.5e-10,2.0e-10]
xlim = [0,0.5]
    
fig3, ax3 = plt.subplots()
color = 'tab:blue'
ax3.set_ylabel('Gradiometer',color=color)
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_xlabel('Time (s)')
ax3.set_ylim(ylim)
ax3.set_xlim(xlim)
ax3.grid(True)

# ax3.plot(time,grad_data,'b')
ax3.plot(time,grad_data_bl[0],'b')


color = 'tab:green'
ax4 = ax3.twinx()
ax4.set_ylim(ylim)
ax4.set_ylabel('Magnetometer',color=color)
ax4.tick_params(axis='y', labelcolor=color)
ax4.grid(True)

# ax4.plot(time,-1*mag_data,color=color)
ax4.plot(time,-1*mag_data_bl[0],color=color)
 
ind = list()
chop = [round(0.45*sr),round(0.75*sr)]

# fig,ax = plt.subplots()

# data_g.plotpower(fig,ax)
    
    
 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

