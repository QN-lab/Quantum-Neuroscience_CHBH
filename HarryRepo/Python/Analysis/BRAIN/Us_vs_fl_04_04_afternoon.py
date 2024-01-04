# -*- coding: utf-8 -*-
"""
@author: hxc214
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
# mpl.use('Qt5Agg')
import math
# import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd
import mne
import os
import os.path as op
from os import listdir
from os.path import isfile, join

#%% Our sensor

ch_names = ['chunk', 'value', 'time', 'B_T (pT)', 'error_deg','Aux1_v','Aux2_v','Trig_in2','Demod_X', 'Demod_Y','Stim','B_T_cal']
ch_types_g = ['misc', 'misc', 'misc','misc', 'misc','misc','misc','misc', 'misc', 'misc','stim','mag']

def process_NMOR(files2,ch_names,ch_types_g,path2):
    sfreq_g=837.1

    evoked2 = list()
    data_NM = list()
    times = list()

    for i in range(len((files2))):
        os.chdir(path2 + files2[i])

        data_g=pd.read_csv('_f.csv',sep=',')
        print(data_g.head(3))
        data_g.shape
    
        scal_fac=10
    
        data_g['B_T_cal']=data_g['B_T (pT)']*scal_fac*1e-12
        data_g['Trig_in2']=data_g['Trig_in2'].round(0).astype(int)
        data_g['Aux1_v']=data_g['Aux1_v'].round(0).astype(int)

        data_raw_g=data_g.T
    
        info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types_g, sfreq=sfreq_g)
        raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)
    
        events = mne.find_events(raw_g, stim_channel=['Stim'], min_duration=0.01, mask_type='not_and', \
                                        mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)
        
        epochs = mne.Epochs(raw_g,
          events.astype(int), event_id = 3,
          tmin= -0.4 , tmax=1,
          baseline=(-0.2,0),
          proj=True,
          picks = 'all',
          detrend = 1,
          #reject=reject,
          reject_by_annotation=True,
          preload=True,
          verbose=True)

        evoked2_i = epochs.copy().average(method='mean').filter(2, 35).crop(-0.2,1)
    
        evoked2.append(evoked2_i)
        data_NM.append(evoked2[i].get_data())
        times.append(evoked2[i].times) #+0.500) # Move every time point forward by ???? because we triggered on the negative edge, epoched???
    
    data_grad2 = np.zeros((len(data_NM),len(data_NM[0][0,:])))

    for i in range(len(data_NM)):
        data_grad2[i,:] = data_NM[i][0,:]

    return data_grad2,times
#%% Process and plot

path2 = 'Z:\\Data\\2023_04_04_Zurich_Brain\\main\\'
os.chdir(path2)

files2_grad = next(os.walk('.'))[1]

evoked2_grad = process_NMOR(files2_grad,ch_names,ch_types_g,path2)

path2_dummy = 'Z:\\Data\\2023_04_04_Zurich_Brain\\dummy\\'
os.chdir(path2_dummy)

files2_dummy = next(os.walk('.'))[1]

evoked2_dummy = process_NMOR(files2_dummy,ch_names,ch_types_g,path2_dummy)

fig, axs = plt.subplots(2)
fig.tight_layout(pad=1.5)

for i in range(len(files2_grad)):
    axs[0].plot(evoked2_grad[1][i],evoked2_grad[0][i,:],label = str(files2_grad[i]))
    
for i in range(len(files2_dummy)):
    axs[1].plot(evoked2_dummy[1][i],evoked2_dummy[0][i,:],label = str(files2_dummy[i]))

axs[0].set_ylim([-3e-12,3e-12])
axs[1].set_ylim([-3e-12,3e-12])
    
axs[1].set_xlabel('Time (s)')

axs[0].set_ylabel('Field (T)')
axs[1].set_ylabel('Field (T)')

axs[0].set_title('Brain Runs')
axs[1].set_title('Dummy Runs')

axs[0].legend(fontsize=8,loc='upper right')
axs[1].legend(fontsize=8,loc='upper right')

plt.figure()
plt.plot(evoked2_grad[1][0],np.mean(evoked2_grad[0],axis = 0))



