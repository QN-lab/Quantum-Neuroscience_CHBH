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

path = 'Z:\\jenseno-opm\\Data\\2023_04_03_FL_Brain\\'
files = [f for f in listdir(path) if isfile(join(path, f))]

ch_names_fl = ['time',  'FL0101-BZ_CL', 'FL0102-BZ_CL', 'FL0103-BZ_CL', 'FL0104-BZ_CL', 'Input1']
ch_types_fl = ['misc','mag','mag','mag','mag','stim']

info_fl = mne.create_info(ch_names=ch_names_fl, ch_types=ch_types_fl, sfreq=sfreq)

os.chdir(path)

raws = list()
# events_i = list()
events = list()
epochs = list()
evoked = list()
data = list()

for i in range(len(files)):
    
    fl_file_name= files[i]
    raw_fl=mne.io.read_raw_fif(fl_file_name,preload=True)
    info = mne.io.read_info(fl_file_name)
    data_fl=raw_fl.to_data_frame()
    
    data_fl.iloc[:,1:5]=data_fl.iloc[:,1:5]*1e-15 #not working???
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
    evoked.append(epochs[i].copy().average(method='mean').filter(2, 35).crop(-0.2,0.8))
    data.append(evoked[i].get_data())

data_grad = np.zeros((len(data),len(data[0][:,0]),len(data[0][0,:])))

for i in range(len(data)):
    data_grad[i,:,:] = data[i]/1e-12

#%% Functions for next bit

def plot_ERP(epochs):
    evoked_aud = epochs.copy().average(method='mean').filter(3, 15).crop(-0.1,0.8)
    x_data = evoked_aud.get_data()[0]    
    temp = epochs.copy().filter(1, 30).crop(-0.1,0.8).get_data()[:,3,:]
    avg = np.mean(temp,axis = 0) 
    std = np.std(temp,axis = 0) /np.sqrt(temp.shape[0])
    time = evoked_aud.times
    fig1=plt.figure()
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.axvline(.1, color='k', linestyle='--')
    #plt.plot(time,avg)
    plt.plot(time,x_data)
    plt.fill_between(time, x_data-std, x_data+std, color='k', alpha=0.2)
    # plt.title(method)
    # filename_fig = op.join(path_to_save, 'AEF_' + method + '.png')
    # fig1.savefig(filename_fig, dpi=600)

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
    
        scal_fac=28
    
        data_g['B_T_cal']=data_g['B_T (pT)']*scal_fac*1e-12
        data_g['Trig_in2']=data_g['Trig_in2'].round(0).astype(int)
        data_g['Aux1_v']=data_g['Aux1_v'].round(0).astype(int)

        data_raw_g=data_g.T
    
        info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types_g, sfreq=sfreq_g)
        raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)
    
        events = mne.find_events(raw_g, stim_channel=['Stim'], min_duration=0.01, mask_type='not_and', \
                                        mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step',consecutive=True)
    
        raw_list = list()
        events_list = list()
        raw_list.append(raw_g)
        events_list.append(events)
        raw, events = mne.concatenate_raws(raw_list,events_list=events_list)
        del raw_list
        
        epochs = mne.Epochs(raw,
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

        evoked2_i = epochs.copy().average(method='mean').filter(2, 35).crop(-0.8,1)
    
        evoked2.append(evoked2_i)
        data_NM.append(evoked2[i].get_data())
        times.append(evoked2[i].times) #+0.500) # Move every time point forward by ???? because we triggered on the negative edge, epoched???
    
    data_grad2 = np.zeros((len(data_NM),len(data_NM[0][0,:])))

    for i in range(len(data_NM)):
        data_grad2[i,:] = data_NM[i][0,:]

    return data_grad2,times
#%% Process and plot

path2 = 'Z:\\jenseno-opm\\Data\\2023_04_03_Zurich_Brain\\'
os.chdir(path2)

files_2i = next(os.walk('.'))[1]
indices = [1,2,3,4] # 6 is Mag

files2_grad = [files_2i[index] for index in indices]

evoked2_grad = process_NMOR(files2_grad,ch_names,ch_types_g,path2)


path2_dummy = 'Z:\\jenseno-opm\\Data\\2023_04_03_Zurich_Brain\\dummy\\'
os.chdir(path2_dummy)

files2_dummyi = next(os.walk('.'))[1]

indices2 = [1,2,3]

files2_dummy = [files2_dummyi[index] for index in indices2]

evoked2_dummy = process_NMOR(files2_dummy,ch_names,ch_types_g,path2_dummy)


plt.figure()

for i in range(len(files)):
    for j in range(0,4):
        plt.plot(evoked[i].times,data_grad[i,j,:],label = 'FL',color='blue') 

plt.xlabel('time (s)')
plt.ylabel('Field (T)')
plt.title('Fieldline Response (all brain runs, all sensors)')

fig, axs = plt.subplots(2)
fig.tight_layout(pad=1.5)

for i in range(len(files2_grad)):
    axs[0].plot(evoked2_grad[1][i],evoked2_grad[0][i,:],label = str(files2_grad[i]))
    
for i in range(len(files2_dummy)):
    axs[1].plot(evoked2_dummy[1][i],evoked2_dummy[0][i,:],label = str(files2_dummy[i]))

axs[0].set_ylim([-2e-12,2e-12])
axs[1].set_ylim([-2e-12,2e-12])
    
axs[1].set_xlabel('Time (s)')

axs[0].set_ylabel('Field (T)')
axs[1].set_ylabel('Field (T)')

axs[0].set_title('Brain Runs')
axs[1].set_title('Dummy Runs')

axs[0].legend(fontsize=8,loc='upper right')
axs[1].legend(fontsize=8,loc='upper right')

plt.figure()
plt.plot(evoked2_grad[1][0],np.mean(evoked2_grad[0],axis = 0))



