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

path = 'Z:\\jenseno-opm\\Data\\2023_03_30_Brain_FL\\'
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

def discard_event_w_T(events_all):
    """discard_event_w_T
    returs events structure for trial without tempreture trigger 
    """
    search_pattern = [1, 0, 1]
    ind = np.where((events_all[:-2,2]==search_pattern[0]) & (events_all[1:-1,2]==search_pattern[1]) & (events_all[2:,2]==search_pattern[2]))
    events_clean = np.concatenate((events_all[ind,0], np.zeros((1,len(ind[0]))),events_all[ind,2]),axis=0).T
    events_id={'start':1}
    return events_clean,events_id

threshold = 1e-12 #T
delta_T = 0.1     #sec
def annotate_drift_epochs(epochs,delta_T,sfreq,threshold,events_clean):
    """annotate_drift_epochs
    makes event = 1 for epochs with drifts lower then threshold 
    makes event = 2 for epochs with drifts over then threshold 
    drift = averaged value of FIRST delta_T sec of epoch - averaged value of LAST delta_T sec of epoch
    """
    ep_data = epochs.get_data()[:,3,:]
    Npoints = round(delta_T * sfreq)
    ind=np.where((np.mean(ep_data[:,0:Npoints],axis=1)-np.mean(ep_data[:,-Npoints:],axis=1))>threshold)
    ##to plot all bad trials 
    #for i,ii in enumerate(ind[0]):
    #    fig=plt.figure(i)
    #    plt.plot(epochs.times,ep_data[ii,:])
    #    plt.axhline(np.mean(ep_data[ii,0:Npoints]), color='k', linestyle='--')
    #    plt.axhline(np.mean(ep_data[ii,-Npoints:]), color='k', linestyle='--')
    events_w_drift = events_clean.copy()
    events_w_drift[ind,2] = 2
    events_id = {'no_drift':1, 'drift':2} 
    return events_w_drift,events_id

def proccess_events(events_all,threshold,delta_T,raw_g):
    """proccess_events
    both discard_event_w_T + annotate_drift_epochs
    """
    events_clean,events_id = discard_event_w_T(events_all)
    raw_list = list()
    events_list = list()
    raw_list.append(raw_g)
    events_list.append(events_clean)
    raw, events = mne.concatenate_raws(raw_list,events_list=events_list)
    del raw_list
    epochs = mne.Epochs(raw,
        events.astype(int), events_id,
        tmin=-0.20 , tmax=1,
        baseline=(-0.2,0),
        proj=True,
        picks = 'all',
        detrend = 1,
        #reject=reject,
        reject_by_annotation=True,
        preload=True,
        verbose=True)

    events_w_drift,events_id = annotate_drift_epochs(epochs,delta_T,sfreq,threshold,events_clean)
    return events_w_drift,events_id

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
path2 = 'Z:\\jenseno-opm\\Data\\2023_03_30_Brain_Zurich\\Copy\\'
os.chdir(path2)

# files_2i = next(os.walk('.'))[1]
files2 = next(os.walk('.'))[1]
# indices = [0,2,4,5] # 6 is Mag

# files2 = [files_2i[index] for index in indices]

ch_names = ['chunk', 'value', 'time', 'B_T (pT)', 'error_deg','Aux1_v','Aux2_v','Trig_in2','Demod_X', 'Demod_Y','B_T_cal']

ch_types_g = ['misc', 'misc', 'misc','misc', 'misc','stim','misc','stim', 'misc', 'misc','mag']

sfreq_g=837.1
print('samplerate: ' + str(sfreq))

def do_the_thing(data_raw_name):
    data_g=pd.read_csv(data_raw_name,sep=',')
    print(data_g.head(3))
    data_g.shape
    
    scal_fac=28
    #this is to create events
    
    data_g['B_T_cal']=data_g['B_T (pT)']*scal_fac*1e-12
    data_g['Trig_in2']=data_g['Trig_in2'].round(0).astype(int)
    data_g['Aux1_v']=data_g['Aux1_v'].round(0).astype(int)

    data_raw_g=data_g.T
    
    info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types_g, sfreq=sfreq_g)
    raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)
    
    events_all = mne.find_events(raw_g, stim_channel=['Aux1_v','Trig_in2'], min_duration=0.01, mask_type='not_and', \
                                        mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step',consecutive=True)
    
    events,events_id = proccess_events(events_all,threshold,delta_T,raw_g)
    raw_list = list()
    events_list = list()
    raw_list.append(raw_g)
    events_list.append(events)
    raw, events = mne.concatenate_raws(raw_list,events_list=events_list)
    del raw_list
    
    epochs = mne.Epochs(raw,
          events.astype(int), event_id = events_id,
          tmin= -0.4 , tmax=1, # normally -0.4 to 1
          baseline=(-0.2,0),
          proj=True,
          picks = 'all',
          detrend = 1,
          #reject=reject,
          reject_by_annotation=True,
          preload=True,
          verbose=True)

    evoked_aud_no_drift = epochs['no_drift'].copy().average(method='mean').filter(2, 35).crop(-0.3,1)
    
    return evoked_aud_no_drift, epochs

evoked2 = list()
data_NM = list()
times = list()
for i in range(len((files2))):
    os.chdir(path2 + files2[i])
    
    evoked2.append(do_the_thing('_f.csv'))
    data_NM.append(evoked2[i][0].get_data())
    times.append(evoked2[i][0].times) #+0.500) # Move every time point forward by ???? because we triggered on the negative edge, epoched???
    
data_grad2 = np.zeros((len(data_NM),len(data_NM[0][0,:])))

for i in range(len(data_NM)):
    data_grad2[i,:] = data_NM[i][0,:]

#%% Plotting

plt.figure()

for i in range(len(files)):
    for j in range(0,4):
        plt.plot(evoked[i].times,data_grad[i,j,:],label = 'FL grad',color='blue') 

plt.xlabel('time (s)')
plt.ylabel('Field (T)')
plt.title('Fieldline Response (all runs, all sensors)')
  
        
plt.figure()
for i in range(len(files2)):
    plt.plot(evoked2[i][0].times,data_grad2[i,:],label = str(files2[i]))
    
plt.xlabel('time (s)')
plt.ylabel('Field (T)')
plt.title('Gradiometer comparsion with 28 Gain factor')
plt.legend()






