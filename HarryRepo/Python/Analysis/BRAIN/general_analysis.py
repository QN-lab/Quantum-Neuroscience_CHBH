# -*- coding: utf-8 -*-
"""
Process all NMOR and/or FL within a given folder path 

@author: hxc214
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('classic')
# mpl.use('Qt5Agg')
# import Harry_analysis as HCA
import pandas as pd
import mne
import os
from os import listdir
from os.path import isfile, join

#%% Functions for sensor type

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
    
        scal_fac=1 #can't remember
    
        data_g['B_T_cal']=data_g['B_T (pT)']*scal_fac*1e-12
        data_g['Trig_in2']=data_g['Trig_in2'].round(0).astype(int)
        data_g['Aux1_v']=data_g['Aux1_v'].round(0).astype(int)

        data_raw_g=data_g.T
    
        info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types_g, sfreq=sfreq_g)
        raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)
        # raw_gf = raw_g.copy().filter(l_freq=1, h_freq=35).notch_filter(50)    
        events = mne.find_events(raw_g, stim_channel=['Stim'], min_duration=0.01, mask_type='not_and', \
                                        mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)
        reject_criteria = dict(mag=1e-12)
        epochs = mne.Epochs(raw_g,
          events.astype(int), event_id = 3,
          tmin= -0.2 , tmax=1,
          baseline=(-0.2,0),
          proj=True,
          picks = 'mag',
          detrend = 1,
          reject = reject_criteria, 
          reject_by_annotation=True,
          preload=True,
          verbose=True)

        evoked2_i = epochs.copy().average(method='mean').crop(-0.2,1)
    
        evoked2.append(evoked2_i)
        data_NM.append(evoked2[i].get_data())
        times.append(evoked2[i].times) 
        
    data_grad2 = np.zeros((len(data_NM),len(data_NM[0][0,:])))

    for i in range(len(data_NM)):
        data_grad2[i,:] = data_NM[i][0,:]

    return data_grad2, times, evoked2


#%% Process and plot

#Zurich Setup channels
ch_names = ['chunk', 'value', 'time', 'B_T (pT)', 'error_deg','Aux1_v','Aux2_v','Trig_in2','Demod_X', 'Demod_Y','Stim','B_T_cal']
ch_types_g = ['misc', 'misc', 'misc','misc', 'misc','misc','misc','misc', 'misc', 'misc','stim','mag']

#Read in all files within folder
path_NM = 'Z:\\jenseno-opm\\Data\\2023_05_04_Zurich_dipole\\runs\\'
os.chdir(path_NM)
files_NM = next(os.walk('.'))[1]

#Process each run and output
evoked_NM = process_NMOR(files_NM,ch_names,ch_types_g,path_NM)

#Plot our sensor
plt.figure()
for i in range(len(files_NM)):
    plt.plot(evoked_NM[1][i],evoked_NM[0][i,:],label = str(files_NM[i]))

plt.legend(fontsize = 7, loc = 'lower right')
plt.title('Our Sensor')
plt.xlabel('Time (s)')
plt.ylabel('Calculated Field (T)')
plt.grid()

#TFR plot
freqs = np.arange(2, 31, 1)
n_cycles = freqs/2
time_bandwidth = 2.0

power_g = mne.time_frequency.tfr_multitaper(evoked_NM[2][3], n_cycles=n_cycles, return_itc=False, use_fft=True, freqs=freqs, average=True, decim=3)
power_g.plot(['B_T_cal'])

#%% FL stuff

def process_FL(files_FL,info_FL):
    raws = list()
    events = list()
    epochs = list()
    evoked = list()
    data = list()
    times = list()

    for i in range(len(files_FL)):
    
        fl_file_name= files_FL[i]
        raw_fl=mne.io.read_raw_fif(fl_file_name,preload=True)
        data_fl=raw_fl.to_data_frame()
    
        # data_fl.iloc[:,1]=data_fl.iloc[:,1]*1e-15 #not working???
        data_fl.iloc[:,1:5]=data_fl.iloc[:,1:5]*1e-15
        data_fl['Input1']=data_fl['Input1'].round(0).astype(int)
        data_fl['Input1']=data_fl['Input1'].replace(1,2,inplace=False)
        raw_data_fl=data_fl.T
        data_fl=raw_fl.to_data_frame()
    
        raw_fl= mne.io.RawArray(raw_data_fl, info_FL, verbose=True)
        # raw_flf = raw_fl.copy().filter(l_freq=1, h_freq=35).notch_filter(50)
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
        evoked.append(epochs[i].copy().average(method='mean').filter(2, 35).crop(-0.2,1))
        data.append(evoked[i].get_data())
        times.append(evoked[i].times)
        
    data_grad = np.zeros((len(data),4,len(data[0][0,:])))

    for i in range(len(data)):
        data_grad[i,:,:] = data[i] # Raw data format: [runs,sensors,data]

    return data_grad, times, evoked 



#%% LOAD FIELDLINE
sfreq = 1000

path_FL = 'Z:\\jenseno-opm\\Data\\2023_05_04_FL_dipole\\1\\'
files_FL = [f for f in listdir(path_FL) if isfile(join(path_FL, f))]
os.chdir(path_FL)

ch_names_fl = ['time','FL0101-BZ_CL','FL0102-BZ_CL','FL0103-BZ_CL','FL0104-BZ_CL', 'Input1']
ch_types_fl = ['misc','mag','mag','mag','mag','stim']

info_FL = mne.create_info(ch_names=ch_names_fl, ch_types=ch_types_fl, sfreq=sfreq)

#Process each run and output
evoked_FL = process_FL(files_FL,info_FL)

#Plot our sensor
plt.figure()
for i in range(len(files_FL)):
    plt.plot(evoked_FL[1][i],np.squeeze(evoked_FL[0][i,1,:]),label = str(files_FL[i]))

plt.legend(fontsize = 7, loc = 'lower right')
plt.title('Fieldline')
plt.xlabel('Time (s)')
plt.ylabel('Calculated Field (T)')
plt.grid()

#TFR plot
freqs = np.arange(2, 31, 1)
n_cycles = freqs/2
time_bandwidth = 2.0

# power_g = mne.time_frequency.tfr_multitaper(evoked_NM[2][3], n_cycles=n_cycles, return_itc=False, use_fft=True, freqs=freqs, average=True, decim=3)
# power_g.plot(['B_T_cal'])
