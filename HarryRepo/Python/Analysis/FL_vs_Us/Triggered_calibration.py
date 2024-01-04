# -*- coding: utf-8 -*-
"""
Created on Fri M24 10:00 2023

@author: H
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
import math
# import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd
import mne
import os

#Functions

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
delta_T = 0.1 #sec

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

def proccess_events(events_all,threshold,delta_T):
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


#%%Load Zurich
base_dir = 'Z:\\jenseno-opm\\Data\\2023_03_23_calibration\\'

folder_idx = [10,12,20,30,40,120] #pTpp that exist for both mag and grad    
sfreq = 837.1

ch_names = ['chunk', 'Shift Hz', 'time s', 'B_T (pT)', 'Aux1_v','Trig_in2','Demod_X', 'Demod_Y']
ch_types = ['misc', 'misc', 'misc','mag', 'stim','stim', 'misc', 'misc']

info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

scal_fac=1

z_m_list = list([])
z_g_list = list([])
z_m_data = list([])
z_g_data = list([])

raw_m = list([])
raw_g = list([])

events_m = list([])
events_g = list([])

for i in range(len(folder_idx)):
    z_m_list.append('dipole_{}pTpp_mag_000'.format(folder_idx[i]))
    z_g_list.append('dipole_{}pTpp_grad_000'.format(folder_idx[i]))
    
    os.chdir(base_dir+z_m_list[i])
    z_m_data.append((pd.read_csv('dipole_{}pTpp_mag_000_f.csv'.format(folder_idx[i]),sep=',')).T)
    
    os.chdir(base_dir+z_g_list[i])
    z_g_data.append((pd.read_csv('dipole_{}pTpp_grad_000_f.csv'.format(folder_idx[i]),sep=',')).T)
    
    
    raw_m.append(mne.io.RawArray(z_m_data[i], info, verbose=True))
    raw_g.append(mne.io.RawArray(z_g_data[i], info, verbose=True))
    
    events_m.append(mne.find_events(raw_m[i], stim_channel=['Aux1_v','Trig_in2'], min_duration=0.001001, mask_type='not_and', \
                                    mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step',consecutive=True))
        
    events_g.append(mne.find_events(raw_g[i], stim_channel=['Aux1_v','Trig_in2'], min_duration=0.001001, mask_type='not_and', \
                                    mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step',consecutive=True))







