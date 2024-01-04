# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:08:08 2023

@author: hxc214
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

#%% LOAD FIELDLINE
folder_fl = 'Z:/Data/2023_02_23_FL/1/20230223/1/'
fname_fl = '20230223_175916_1_1_square_1Hz_100pTcm_dBz_2_raw.fif'

raw = mne.io.read_raw_fif(folder_fl+fname_fl)
info = mne.io.read_info(folder_fl+fname_fl)
print(info)

df=raw.to_data_frame()
print(df.head(3))
df.shape

df0=df
df0['00:03-BZ_CL']=df['00:03-BZ_CL']*1e-15
df0['00:05-BZ_CL']=df['00:05-BZ_CL']*1e-15
df0['Syn_grad']=(df0['00:03-BZ_CL']-df0['00:05-BZ_CL'])#/0.04
print(df0.head(3))

# df0.to_csv(Folder+fname+'_syn_grad.csv', sep=',', index=False)

events = mne.find_events(raw,stim_channel='Input-1',min_duration= 100/raw.info['sfreq'])

ch_m1=['00:03-BZ_CL']
ch_m2=['00:05-BZ_CL']
ch_m=ch_m1+ch_m2
ch_sg=['Syn_grad']

scaling = dict(mag=1e9)

epochs = mne.Epochs(raw, events,tmin=0.255, tmax = 6, baseline=None, preload=True)

# epochs.plot(n_epochs=1,scalings = scaling)


evoked=epochs.average()

times = evoked.times-0.255 # remove trigger

qq = evoked.get_data()

grad_data_fl = qq[1,:] - qq[0,:]

#Plotting
fig, ax1 = plt.subplots()
color = 'tab:orange'
ax1.plot(times[:4000],grad_data_fl[:4000],label = 'synth',color=color)
ax1.set_ylabel('deltaB(nT)', color = color) 


#%%Load Zurich

csv_sep = ';'

def ReadData(folder_path,filename,headername,csv_sep):
    out_sig = pd.read_csv(folder_path+filename,sep=csv_sep)
    out_headers = pd.read_csv(folder_path+headername,sep=csv_sep)
    return out_sig, out_headers

def DAQ_read_shift(folder_path,csv_sep):
    
    filename = 'dev3994_pids_0_stream_shift_avg_00000.csv'
    headername = 'dev3994_pids_0_stream_shift_avg_header_00000.csv'

    out_sig, out_headers = ReadData(folder_path, filename, headername, csv_sep)

    return out_sig, out_headers

class DAQ_Trigger:
    def __init__(self,sig,header):
        
        self.sig=sig.drop_duplicates(keep='last')
        self.header=header
        
        #Pull from headers
        self.ChunkSize = self.header['chunk_size'].tolist() #Size of each chunk
        self.patch = self.header['chunk_number'].tolist() # Prints a list of the number of runs
        self.run_names = self.header['history_name'].tolist()
        # self.filter_BW = self.header['history_name'].tolist()
        
        #Pull from Signals
        self.Chunk = self.sig['chunk'].tolist()
        self.timestamps = np.array(self.sig['timestamp'].tolist())
        self.Data = self.sig['value'].tolist()
        
        #Organise by Run
        self.chunked_data = np.array(self.Data).reshape(len(self.patch),self.ChunkSize[0])
        chunked_timestamps = self.timestamps.reshape(len(self.patch),self.ChunkSize[0])
        
        #WHEN DOING BRAIN STUFF NEED TO THINK ABOUT THIS. 
        self.chunked_time = np.zeros(chunked_timestamps.shape)
        for i in range(len(self.patch)):
            self.chunked_time[i,:] = (chunked_timestamps[i,:] - chunked_timestamps[i,0])/60e6
            
class DAQ_Tracking_PURE(DAQ_Trigger):
        def __init__(self,sig,header):
            DAQ_Trigger.__init__(self, sig, header) #Share init conditions from Parent Class (DAQ_Trigger)
            self.chunked_field = self.chunked_data*0.071488e-9

folder_g = 'Z:/jenseno-opm/Data/2023_02_23_FL/1/20230223/1/'
fname_g = '20230223_175916_1_1_square_1Hz_100pTcm_dBz_2_raw.fif'

daq_g = 'Z:/jenseno-opm/Data/2023_02_23_Zurich/square_1Hz_100pTcm_dBz_2_000/'      
trace_g, trace_legends_g = DAQ_read_shift(daq_g,csv_sep)

traces_g = DAQ_Tracking_PURE(trace_g,trace_legends_g)

grad_data_us = np.mean(traces_g.chunked_data,axis = 0)

ax2 = ax1.twinx()

color = 'tab:cyan'
ax2.plot(traces_g.chunked_time[0,:3348],grad_data_us[:3348],label = 'NMOR',color=color)
ax2.set_ylabel('deltaB (Hz))', color = color) 
plt.show()


