# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:10:30 2023

@author: kowalcau
"""
""
import os
import mne
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.get_option("display.max_columns",15)

base_directory_fl='Z:\\Data\\2023_07_19\\FL\\'

os.chdir(base_directory_fl)

fl_file_name='20230719_H_Heart_raw.fif'

raw_fl=mne.io.read_raw_fif(fl_file_name,preload=True)
info = mne.io.read_info(fl_file_name)
print(info)

data_fl=raw_fl.to_data_frame()
#print(data_fl.head(3))
#change units to T
data_fl.iloc[:,1:3]=data_fl.iloc[:,1:3]*1e-15
#data_fl.iloc[:,0]=data_fl.iloc[:,0]*1e-3
#Analogue input on Fieldline makes it difficult for MNE to find the events so we round the values to the nearest integer 
data_fl['Input1']=data_fl['Input1'].round(0).astype(int)
data_fl['Input1']=data_fl['Input1'].replace(1,2,inplace=False)
print(data_fl.head(3))

data_fl['time']=data_fl['time']
data_fl['Grad1']=data_fl['FL0101-BZ_CL']-data_fl['FL0102-BZ_CL']
#data_fl['Grad2']=data_fl['FL0103-BZ_CL']-data_fl['FL0104-BZ_CL']
print(data_fl.head(3))
sfreq_fl=1/(data_fl['time'].iloc[1])

#ch_names_fl = ['time',   'FL0101-BZ_CL','FL0102-BZ_CL','FL0103-BZ_CL','FL0104-BZ_CL','Input1', 'Stim', 'Grad1', 'Grad2']
# ch_types_fl = ['misc',  'mag', 'mag', 'mag','mag','stim', 'stim', 'mag','mag']

ch_names_fl = ['time',   'FL0101-BZ_CL','FL0102-BZ_CL','Input1', 'Grad1']
ch_types_fl = ['misc',  'mag', 'mag','stim', 'mag']



data_raw_fl=data_fl.T

info_fl = mne.create_info(ch_names=ch_names_fl, ch_types=ch_types_fl, sfreq=sfreq_fl)
raw_fl= mne.io.RawArray(data_raw_fl, info_fl, verbose=True)
#%%

reject_criteria = dict(mag=20e-5)
reject_criteria_f = dict(mag=20e-5)
l_freq_r = 0.01 ##filter settings for the raw #(doesn't do much right now)
h_freq_r =40
l_freq = 0.01 ##fiter settings for evoked data
h_freq = 30
tmin_e = -0.2
tmax_e = 1
baseline =(-0.2,0)
picks_fl = ['FL0101-BZ_CL']

events_fl = mne.find_events(raw_fl, stim_channel=['Input1'], min_duration=0.01, mask_type='not_and', \
                                     mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)
#mne.viz.plot_events(events_all_g,first_samp=3000 ) 

raw_flf = raw_fl.copy().filter(l_freq=l_freq_r,h_freq=h_freq_r)


##plot  to check if the stim chanell is rejecting trials with the temperatrue sensing on
raw_flf.copy().pick_types(meg=True, stim=True).plot( start=0, duration=20, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'), event_color={ 3:'w'})



