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

path = 'Z:\\jenseno-opm\\Data\\2023_03_28_FL\\'
files = [f for f in listdir(path) if isfile(join(path, f))]

regex = 'n_(.*)mVpp'

x = list()
for i in range(len(files)):
    x.append(re.findall(regex, files[i]))

raws = list()
# events_i = list()
events = list()
epochs = list()
evoked = list()
data = list()

for i in range(len(files)):
    raws.append(mne.io.read_raw_fif(path+files[i]))
    events.append(mne.find_events(raws[i],stim_channel='Input1', min_duration= 0.001001))
    epochs.append(mne.Epochs(raws[i],
        events[i].astype(int)[:20],
        event_id=None,
        tmin=-0.20 , tmax=1,
        baseline=(-0.2,0),
        proj=True,
        picks = 'all',
        detrend = 1,
        reject_by_annotation=True,
        preload=True,
        verbose=True))
    evoked.append(epochs[i].copy().average(method='mean').filter(1, 100).crop(-0.1,0.8))
    data.append(evoked[i].get_data())

data_grad = np.zeros((len(data),len(data[0][0,:])))

for i in range(len(data)):
    data_grad[i,:] = (data[i][1,:] - data[i][0,:])/1e-12


    
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
    plt.title(method)
    filename_fig = op.join(path_to_save, 'AEF_' + method + '.png')
    fig1.savefig(filename_fig, dpi=600)

    

#%% Our sensor
path2 = 'Z:\\jenseno-opm\\Data\\2023_03_28_Zurich\\'
os.chdir(path2)

files_2i = next(os.walk('.'))[1]
files2 = files_2i[:2] # First 2 matching files

sfreq=837.1
print('samplerate: ' + str(sfreq))


def do_the_thing(data_raw_name):
    data_g=pd.read_csv(data_raw_name,sep=',')
    print(data_g.head(3))
    data_g.shape
    
    scal_fac=72
    #this is to create events
    
    data_g['B_T (pT)']=data_g['B_T (pT)']*scal_fac
    pd.set_option('display.max_columns', 20)
    data_g.head(10)
    
    ch_names = ['chunk', 'Shift Hz', 'time s', 'B_T (pT)', 'Aux1_v','Trig_in2','Demod_X', 'Demod_Y']
    
    ch_types_g = ['misc', 'misc', 'misc','mag', 'stim','stim', 'misc', 'misc']
    data_raw_g=data_g.T
    
    info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types_g, sfreq=sfreq)
    raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)
    
    events_all = mne.find_events(raw_g, stim_channel=['Aux1_v','Trig_in2'], min_duration=0.001001, mask_type='not_and', \
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
          tmin=-0.40 , tmax=1,
          baseline=(-0.4,0),
          proj=True,
          picks = 'all',
          detrend = 1,
          #reject=reject,
          reject_by_annotation=True,
          preload=True,
          verbose=True)
    


    evoked_aud_no_drift = epochs['no_drift'].copy().average(method='mean').filter(1, 100).crop(-0.1,0.8)
    method = 'no_drift_over_'+str(threshold*1e12)+'pT'
    
    return evoked_aud_no_drift

evoked2 = list()
data_NM = list()

for i in range(len((files2))):
    os.chdir(path2 + files2[i])
    
    evoked2.append(do_the_thing('_f.csv'))
    data_NM.append(evoked2[i].get_data())
    
data_grad2 = np.zeros((len(data_NM),len(data_NM[0][0,:])))

for i in range(len(data_NM)):
    data_grad2[i,:] = data_NM[i][0,:]


#%% Plotting


# for i in range(len(files)):
#     plt.figure()
#     plt.plot(evoked[i].times,data_grad[i,:],label = 'grad')
#     plt.title(files[i])
#     plt.xlabel('time (s)')
#     plt.ylabel('Field (T)')
#     plt.legend()
    
    
# for i in range(len(files2)):
#     plt.figure()
#     plt.plot(evoked2[i].times,data_grad2[i,:],label = 'grad')
#     plt.title(files2[i])
#     plt.xlabel('time (s)')
#     plt.ylabel('Field (pT)')
#     plt.legend()

plt.figure()
for i in range(len(files)):
    if i == 0:
        plt.plot(evoked[i].times,data_grad[i,:],label = 'FL grad',color='blue')
        plt.plot(evoked2[i].times,data_grad2[i,:],label = 'NMOR grad',color='green')
    
    if i > 0:
        plt.plot(evoked[i].times,data_grad[i,:],color='blue')
        plt.plot(evoked2[i].times,data_grad2[i,:],color='green')
        
plt.xlabel('time (s)')
plt.ylabel('Field (pT)')
plt.title('Gradiometer comparsion with 72 Gain factor')
plt.legend()






