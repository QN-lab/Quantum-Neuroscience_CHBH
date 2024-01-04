# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:59:52 2023

@author: kowalcau
"""


import os

import mne
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.get_option("display.max_columns",15)

#Z:\Data\Processing_directory\Attenuation_gradiometer
base_directory='Z:\\Data\\2023_07_11_FL_AER'
dir_name = os.path.basename(base_directory)
os.chdir(base_directory)
print(dir_name)
file_name='20230711_161222_C4B4_birmingham_AER01_raw.fif'
raw=mne.io.read_raw_fif(file_name,preload=True)
info = mne.io.read_info(file_name)
print(info)

data_fl=raw.to_data_frame()
data_fl.head(3)
data_fl.shape
data_fl.iloc[:,1:17]=data_yes
data_fl['Input1']=(data_fl['Input1']).round(0).astype(int)


#df0.to_csv(file_name+'_syn_grad.csv', sep=',', index=False)

ch_names_fl = ['time', 'FL0101-BZ_CL', 'FL0102-BZ_CL', 'FL0103-BZ_CL', 'FL0104-BZ_CL','FL0105-BZ_CL', 'FL0106-BZ_CL', 
               'FL0107-BZ_CL', 'FL0108-BZ_CL', 'FL0109-BZ_CL', 'FL0110-BZ_CL', 'FL0111-BZ_CL', 'FL0112-BZ_CL', 
               'FL0113-BZ_CL', 'FL0114-BZ_CL', 'FL0115-BZ_CL', 'FL0116-BZ_CL', 'Input1']
ch_types_fl = ['misc', 'mag', 'mag','mag', 'mag','mag', 'mag','mag', 'mag','mag', 'mag','mag', 'mag','mag', 'mag','mag',
               'mag', 'stim']

data_fl=data_fl.T

sfreq_fl =  1000.0 #in Hz

info_fl = mne.create_info(ch_names=ch_names_fl, ch_types=ch_types_fl, sfreq=sfreq_fl)
raw_fl= mne.io.RawArray(data_fl, info_fl, verbose=True)

###################################
reject_criteria = dict(mag=30e-12)
reject_criteria_f = dict(mag=30e-12)
l_freq_r = 0.01 ##filter settings for the raw #(doesn't do much right now)
h_freq_r =400
l_freq = 0.01 ##fiter settings for evoked data
h_freq = 30
tmin_e = -0.2
tmax_e = 1
baseline =(-0.2,0)
picks_fl = ['mag']




# events_all_fl = mne.find_events(raw_fl, stim_channel=['Input1'], min_duration=0.01,  mask_type='not_and', \
#                                 mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)

events_m_fl = mne.find_events(raw, stim_channel=['Input1'], min_duration=0.05,   mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset', consecutive=True)

raw.copy().pick_types(meg=True, stim=True).plot(events=events_m_fl, start=0, duration=25, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'), event_color={ 1:'b',2: 'g'})
mne.viz.plot_events(events_m_fl) 


raw_flf = raw_fl.copy().filter(l_freq=l_freq_r, h_freq=h_freq_r)

epochs_flf=mne.Epochs(raw_flf,events=events_m_fl, picks=picks_fl, tmin=tmin_e,tmax=tmax_e, baseline =baseline, reject=reject_criteria_f,preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)
evoked_flf=epochs_flf.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)
evoked_flf.plot()
epochs_N=mne.Epochs(raw_flf,events=events_m_fl, event_id= 2, picks=picks_fl, tmin=tmin_e,tmax=tmax_e, baseline =baseline, reject=reject_criteria_f,preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)
evoked_N=epochs_N.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)
evoked_N.plot()
epochs_O=mne.Epochs(raw_flf,events=events_m_fl, event_id= 1, picks=picks_fl, tmin=tmin_e,tmax=tmax_e, baseline =baseline, reject=reject_criteria_f,preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)
evoked_O=epochs_O.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)
evoked_O.plot()


N_events_fl = len(epochs_flf)#events_m_fl.shape[0]
epochs_N=
np.shape(evoked_flf)
#




# #evoked_fl_exp = evoked_flf.copy().pick_types(meg=True,stim=False)
