# -*- coding: utf-8 -*-
"""
@author: hxc214
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('classic')
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
    sfreq_g = 837.1

    evoked2 = list()
    data_NM = list()
    times = list()

    for i in range(len((files2))):
        os.chdir(path2 + files2[i])

        data_g=pd.read_csv('_f.csv',sep=',')
        print(data_g.head(3))
        data_g.shape
    
        scal_fac=27 #can't remember
    
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
          reject_by_annotation=True,
          preload=True,
          verbose=True)

        evoked2_i = epochs.copy().average(method='mean').filter(2, 25).crop(-0.2,1)
    
        evoked2.append(evoked2_i)
        data_NM.append(evoked2[i].get_data())
        times.append(evoked2[i].times) 
        
    data_grad2 = np.zeros((len(data_NM),len(data_NM[0][0,:])))

    for i in range(len(data_NM)):
        data_grad2[i,:] = data_NM[i][0,:]

    return data_grad2, times, evoked2
#%% Process and plot

path2 = 'Z:\\jenseno-opm\\Data\\2023_04_19_Zurich_Brain\\runs\\'
os.chdir(path2)

files2_grad = next(os.walk('.'))[1]

evoked2_grad = process_NMOR(files2_grad,ch_names,ch_types_g,path2)

#Our sensor
plt.figure()
for i in range(len(files2_grad)):
    plt.plot(evoked2_grad[1][i],evoked2_grad[0][i,:],label = str(files2_grad[i]))

plt.legend(fontsize = 7, loc = 'lower right')
plt.title('Our Sensor')
plt.xlabel('Time (s)')
plt.ylabel('Calculated Field (T)')
plt.grid()


freqs = np.arange(2, 31, 1)
n_cycles = freqs/2
time_bandwidth = 2.0

power_g = mne.time_frequency.tfr_multitaper(evoked2_grad[2][3], n_cycles=n_cycles, return_itc=False, use_fft=True, freqs=freqs, average=True, decim=3)
power_g.plot(['B_T_cal'])

#%% LOAD FIELDLINE
sfreq = 1000

path = 'Z:\\jenseno-opm\\Data\\2023_04_19_FL_Brain\\'
files = [f for f in listdir(path) if isfile(join(path, f))]

ch_names_fl = ['time','FL0101-BZ_CL','FL0102-BZ_CL','FL0103-BZ_CL','FL0104-BZ_CL', 'Input1']
ch_types_fl = ['misc','mag','mag','mag','mag','stim']

info_fl = mne.create_info(ch_names=ch_names_fl, ch_types=ch_types_fl, sfreq=sfreq)

os.chdir(path)

raws = list()
events = list()
epochs = list()
evoked = list()
data = list()

for i in range(len(files)):
    
    fl_file_name= files[i]
    raw_fl=mne.io.read_raw_fif(fl_file_name,preload=True)
    info = mne.io.read_info(fl_file_name)
    data_fl=raw_fl.to_data_frame()
    
    # data_fl.iloc[:,1]=data_fl.iloc[:,1]*1e-15 #not working???
    data_fl.iloc[:,1:5]=data_fl.iloc[:,1:5]*1e-15
    data_fl['Input1']=data_fl['Input1'].round(0).astype(int)
    data_fl['Input1']=data_fl['Input1'].replace(1,2,inplace=False)
    raw_data_fl=data_fl.T
    data_fl=raw_fl.to_data_frame()
    
    raw_fl= mne.io.RawArray(raw_data_fl, info_fl, verbose=True)
    
    raws.append(raw_fl)
    events.append(mne.find_events(raw_fl, stim_channel=['Input1'], min_duration=0.01, mask_type='not_and', \
                                     mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True))
    epochs.append(mne.Epochs(raws[i],
        events[i].astype(int),
        event_id=2,
        tmin=-0.4 , tmax=1,
        baseline=(-0.2,0),
        proj=True,
        picks = 'all',
        detrend = 1,
        reject_by_annotation=True,
        preload=True,
        verbose=True))
    evoked.append(epochs[i].copy().average(method='mean').filter(2, 25).crop(-0.2,1))
    data.append(evoked[i].get_data())

data_grad = np.zeros((len(data),4,len(data[0][0,:])))

for i in range(len(data)):
    data_grad[i,:,:] = data[i]
    
   #Fieldline
plt.figure()
for i in range(len(files)):
    plt.plot(evoked[i].times,np.mean(data_grad[i,:,:],axis=0),label = str(files[i]))

plt.legend(fontsize = 7, loc = 'lower right')
plt.title('Fieldlines')
plt.xlabel('Time (s)')
plt.ylabel('Calculated Field (T)')
plt.grid()

power_fl = mne.time_frequency.tfr_multitaper(evoked[1], n_cycles=n_cycles, return_itc=False, use_fft=True, freqs=freqs, average=True, decim=3)
power_fl.plot(['FL0102-BZ_CL'])
