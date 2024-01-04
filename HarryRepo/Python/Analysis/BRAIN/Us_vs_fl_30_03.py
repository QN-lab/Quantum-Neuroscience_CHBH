# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 18:18:44 2023

@author: kowalcau
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:41:31 2023

@author: kowalcau
"""

# -*- coding: utf-8 -*-
"""
@author: hxc214
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
mpl.use('Qt5Agg')
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

path = 'Z:\\Data\\2023_03_30_Brain_FL\\'
files = [f for f in listdir(path) if isfile(join(path, f))]

raws_fl_list = list()
# events_i = list()
events_fl = list()
epochs = list()
evoked = list()
data = list()

for i in range(len(files)):
    raws_fl_list.append(mne.io.read_raw_fif(path+files[i]))
    events_fl.append(mne.find_events(raws_fl_list[i],stim_channel='Input1', min_duration= 0.001001))
    epochs.append(mne.Epochs(raws_fl_list[i],
        events_fl[i].astype(int)[:20],
        event_id=None,
        tmin=-0.20 , tmax=1,
        baseline=(0.6,0.8),
        proj=True,
        picks = 'all',
        detrend = 1,
        reject_by_annotation=True,
        preload=True,
        verbose=True))
    evoked.append(epochs[i].copy().average(method='mean').filter(2, 35).crop(-0.2,0.8))
    data.append(evoked[i].get_data())

data_grad = np.zeros((len(data),len(data[0][0,:])))

for i in range(len(data)):
    data_grad[i,:] = data[i][0,:]/1e-12


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

threshold = 1e-11 #T
delta_T = 0.1    #sec
def annotate_drift_epochs(epochs,delta_T,sfreq,threshold,events_clean):
    """annotate_drift_epochs
    makes event = 1 for epochs with drifts lower then threshold 
    makes event = 2 for epochs with drifts over then threshold 
    drift = averaged value of FIRST delta_T sec of epoch - averaged value of LAST delta_T sec of epoch
    """
    ep_data = epochs.get_data()[:,8,:]
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
        tmin=-0.2 , tmax=1,
        baseline=(0.6,0.8),
        proj=True,
        picks = 'all',
        detrend = 1,
        #reject=reject,
        reject_by_annotation=True,
        preload=True,
        verbose=True)

    events_w_drift,events_id = annotate_drift_epochs(epochs,delta_T,sfreq,threshold,events_clean)
    return events_w_drift,events_id

# def plot_ERP(epochs):
#     evoked_aud = epochs.copy().average(method='mean').filter(3, 15).crop(-0.1,0.8)
#     x_data = evoked_aud.get_data()[0]    
#     temp = epochs.copy().filter(1, 30).crop(-0.1,0.8).get_data()[:,3,:]
#     avg = np.mean(temp,axis = 0) 
#     std = np.std(temp,axis = 0) /np.sqrt(temp.shape[0])
#     time = evoked_aud.times
#     fig1=plt.figure()
#     plt.axhline(0, color='k', linestyle='--')
#     plt.axvline(0, color='k', linestyle='--')
#     plt.axvline(.1, color='k', linestyle='--')
#     #plt.plot(time,avg)
#     plt.plot(time,x_data)
#     plt.fill_between(time, x_data-std, x_data+std, color='k', alpha=0.2)
#     plt.title(method)
#     filename_fig = op.join(path_to_save, 'AEF_' + method + '.png')
#     fig1.savefig(filename_fig, dpi=600)


#%% Our sensor
path2 = 'Z:\\Data\\2023_03_30_Brain_Zurich\\'
os.chdir(path2)

files_2i = next(os.walk('.'))[1]

indices = [4,5,6,8] # 6 is Mag

files2 = [files_2i[index] for index in indices]

sfreq=837.1
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
    pd.set_option('display.max_columns', 20)
    data_g.head(10)
    
    ch_names = ['chunk', 'value', 'time', 'B_T (pT)', 'error_deg','Aux1_v','Aux2_v','Trig_in2','Demod_X', 'Demod_Y','Stim','B_T_cal']
    
    ch_types_g = ['misc', 'misc', 'misc','eeg', 'misc','misc','stim','stim', 'misc', 'misc','stim','mag']
    data_raw_g=data_g.T
    
    info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types_g, sfreq=sfreq)
    raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)
    
    events_all = mne.find_events(raw_g, stim_channel=['Stim'], min_duration=0.01, mask_type='not_and', \
                                        mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step',consecutive=True)
    
    events,events_id = proccess_events(events_all,threshold,delta_T,raw_g)
    raw_list = list()
    events_list = list()
    raw_list.append(raw_g)
    events_list.append(events)
    raw, events = mne.concatenate_raws(raw_list,events_list=events_list)
    del raw_list
    
    epochs = mne.Epochs(raw,
          events.astype(int), events_id,
          tmin= -0.5 , tmax=0.9, # normally -0.4 to 1
          baseline=(0.6,0.8),
          proj=True,
          picks = 'all',
          detrend = 1,
          #reject=reject,
          reject_by_annotation=True,
          preload=True,
          verbose=True)

    evoked_aud_no_drift = epochs['no_drift'].copy().average(method='mean').filter(2, 35).crop(-0.3,1)
    
    return evoked_aud_no_drift, raw_g, events

evoked2 = list()
data_NM = list()
times = list()
raw_g_list = list()
events_Q = list()

for i in range(len((files2))):
    os.chdir(path2 + files2[i])
    
    evoked2.append(do_the_thing('_f.csv'))
    
    raw_g_list.append(evoked2[i][1])
    data_NM.append(evoked2[i][0].get_data())
    events_Q.append(evoked2[i][2])
    
    times.append(evoked2[i][0].times) #+0.500) # Move every time point forward by ???? because we triggered on the negative edge, epoched???
    
data_grad2 = np.zeros((len(data_NM),len(data_NM[0][0,:])))

for i in range(len(data_NM)):
    data_grad2[i,:] = data_NM[i][0,:]

#%% Plotting

plt.figure()
for i in range(len(files2)):
    
    plt.plot(times[i],data_grad2[i,:],label = str(files2[i]))
    
    
plt.xlabel('time (s)')
plt.ylabel('Field (pT)')
plt.legend()


chs_all=['B_T_cal','Trig_in2']
chs_fl=['FL0101-BZ_CL', 'FL0102-BZ_CL', 'FL0103-BZ_CL', 'FL0104-BZ_CL', 'Input1']

for i in range(len(files2)):
    chan_idxs2 = [raw_g_list[i].ch_names.index(ch) for ch in chs_all]

    raw_g_list[i].plot(order=chan_idxs2, events=events_Q[i], start=50, duration=5, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'), event_color={1: 'r', 2: 'g'})

for i in range(len(files)):
    chan_idxsfl = [raws_fl_list[i].ch_names.index(ch) for ch in chs_fl]

    raws_fl_list[i].plot(order=chan_idxsfl, events=events_fl[i], start=50, duration=5, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'))


plt.plot(evoked[i].times,data_grad[i,:],label = 'FL grad',color='blue')

plt.figure()
for i in range(len(files)):
    if i == 0:
        plt.plot(evoked[i].times,data_grad[i,:],label = 'FL grad',color='blue')
        plt.plot(times[i],data_grad2[i,:],label = 'NMOR grad',color='green')
    
    if i > 0:
        plt.plot(evoked[i].times,data_grad[i,:],color='blue')
        plt.plot(times[i],data_grad2[i,:],color='green')
        
plt.xlabel('time (s)')
plt.ylabel('Field (pT)')
plt.title('Gradiometer comparsion with 28 Gain factor')
plt.legend()






