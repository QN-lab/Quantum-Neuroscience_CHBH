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

#%% LOAD FIELDLINE
sfreq = 1000

path = 'Z:\\jenseno-opm\\Data\\2023_03_23_FL\\1\\'
files = [f for f in listdir(path) if isfile(join(path, f))]

ch_names_fl = ['time','FL0102-BZ_CL', 'Input1']
ch_types_fl = ['misc','mag','stim']

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
    data_fl['Input1']=data_fl['Input1'].round(0).astype(int)
    data_fl['Input1']=data_fl['Input1'].replace(1,2,inplace=False)
    raw_data_fl=data_fl.T
    data_fl=raw_fl.to_data_frame()
    
    raw_fl= mne.io.RawArray(raw_data_fl, info_fl, verbose=True)
    
    raws.append(raw_fl)
    events.append(mne.find_events(raw_fl, stim_channel=['Input1'], min_duration=0.01, mask_type='not_and', \
                                     mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True))
    epochs.append(mne.Epochs(raws[i],
        events[i].astype(int)[20:],
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

data_grad = np.zeros((len(data),len(data[0][0,:])))

for i in range(len(data)):
    data_grad[i,:] = data[i]*1e-15

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
          events.astype(int)[:20], event_id = 3,
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

    return data_grad2, times
#%% Process and plot

path2 = 'Z:\\jenseno-opm\\Data\\2023_03_23_calibration\\grad\\'
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

#Fieldline
plt.figure()
for i in range(len(files2_grad)):
    plt.plot(evoked[i].times,data_grad[i,:],label = str(files[i]))

plt.legend(fontsize = 7, loc = 'lower right')
plt.title('Fieldlines')
plt.xlabel('Time (s)')
plt.ylabel('Calculated Field (T)')
plt.grid()

#%% FFT

#US
fs = 837.1
plt.figure()
for i in range(len(files2_grad)):

    xf,yf = signal.welch(evoked2_grad[0][i,:],fs,nperseg = fs)
    
    plt.plot(xf,10*np.log10(yf),label = str(files2_grad[i]))
    
plt.xlim([0,100])
plt.title('Our Sensor')
plt.ylabel('Relative power (arb)')
plt.xlabel('Freq (Hz)')
plt.grid()
plt.axvline(x=4,c = 'red')
plt.legend(fontsize = 7, loc = 'upper right')

#FLs
fs2 = 1000
plt.figure()
for i in range(len(files2_grad)):


    xf,yf = signal.welch(data_grad[i,:],fs2,nperseg = fs2)
    
    
    plt.plot(xf,10*np.log10(yf),label = str(files[i]))
    
plt.xlim([0,100])
plt.ylabel('Relative power (arb)')
plt.title('Fieldlines')
plt.xlabel('Freq (Hz)')
plt.grid()
plt.axvline(x=4,c = 'red')
plt.legend(fontsize = 7, loc = 'upper right')


#%% Overlap

Runs1 = [3]
Runs2 = [7]
name = ['10pTpp','20pTpp']
plt.figure()
for i in range(0,2):
    plt.plot(evoked2_grad[1][Runs1[i]],evoked2_grad[0][Runs1[i],:],c = 'blue',label = name[i])
    
    plt.plot(evoked[Runs2[i]].times,data_grad[Runs2[i],:],c = 'red',label = name[i])


