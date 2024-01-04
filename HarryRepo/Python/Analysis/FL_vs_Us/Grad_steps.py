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
folder_fl = 'Z:/jenseno-opm/Data/2023_02_27_FL/1/'
fname_fl = '20230227_125658_1_1_DC_grad_calibrate_raw.fif'

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

mne.viz.plot_events(events)

ch_m1=['00:03-BZ_CL']
ch_m2=['00:05-BZ_CL']
ch_m=ch_m1+ch_m2
ch_sg=['Syn_grad']

scaling = dict(mag=1e9)

epochs = mne.Epochs(raw, events[:-1],tmin=0.255, tmax = 9, baseline=None, preload=True)

evoked=epochs.average()

times = evoked.times-0.255 # remove trigger

qq = evoked.get_data()

grad_data_fl = qq[1,:]-qq[0,:]

#Plotting
fig, ax1 = plt.subplots()
color1 = 'tab:orange'
plt.title('Gradiometer response (FL vs NMOR) ')
plt.grid()
ax1.plot(times,grad_data_fl/1e-9,label = 'Fieldline',color=color1)
ax1.set_ylabel('delta B (nT)', color = color1) 
ax1.set_xlabel('time (s)')



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

daq_g = 'Z:/jenseno-opm/Data/2023_02_28_Zurich/grad_calibrate_2_000/'
trace_g, trace_legends_g = DAQ_read_shift(daq_g,csv_sep)

traces_g = DAQ_Tracking_PURE(trace_g,trace_legends_g)

patch = np.zeros(200) # added to line up both sensor timecourses.

grad_data_us_i = np.concatenate([patch, np.mean(traces_g.chunked_data,axis = 0)])

ax2 = ax1.twinx()

inx = 7783

grad_data_us = grad_data_us_i[:inx]

color2 = 'tab:cyan'
ax2.plot(traces_g.chunked_time[0,:inx],grad_data_us,label = 'NMOR',color=color2)
ax2.set_ylabel('delta B (Hz)', color = color2) 


# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

#%%  Extract Field vales 

tincr = (np.array([1,3,5,7,9,11,13,15,17,19,21,23,25,27])*0.3)+0.08
interval = np.array([0.050,0.250])

t_ROI = np.zeros((len(interval),len(tincr)))
ind_FL = np.zeros((len(interval),len(tincr)))
ind_us = np.zeros((len(interval),len(tincr)))

for i in range(len(interval)):
    t_ROI[i,:] = tincr+interval[i]
    for j in range(len(tincr)):
        ind_FL[i,j] = (np.abs(times - t_ROI[i,j])).argmin()
        ind_us[i,j] = (np.abs(traces_g.chunked_time[0,:] - t_ROI[i,j])).argmin() #chunked time should be the same for all runs
   
sind_FL = ind_FL[0,:]
find_FL = ind_FL[1,:]
sind_us = ind_us[0,:]
find_us = ind_us[1,:]

def pull_mean_vals(sind,find,data):
    
    mean_vals = np.zeros(14)
    for i in range(14):
        mean_vals[i] = np.mean(data[int(sind[i]):int(find[i])])
    return mean_vals
        
vals_FL = pull_mean_vals(sind_FL,find_FL,grad_data_fl)
vals_us = pull_mean_vals(sind_us,find_us,grad_data_us)

# for i in range(len(vals_FL)):
    # ax1.axhline(vals_FL[i])
    # ax2.axhline(vals_us[i], c='red')
        
plt.show()

# plt.figure()
# plt.plot(range(len(grad_data_fl[:1000])),grad_data_fl[:1000])#
# plt.show()
# plt.figure()
# plt.plot(range(len(grad_data_us[:837])),grad_data_us[:837])
# plt.show()

#%% Plot mean values


fig2, ax3 = plt.subplots()
color = 'tab:orange'
ax3.plot(range(len(vals_FL)),vals_FL/1e-9,'-o',label='Fieldline',c=color1)
ax3.set_ylabel('delta B (nT)', color = color1) 
plt.xlabel('Field step index')

ax4 = ax3.twinx()
ax4.plot(range(len(vals_us)),vals_us,'-o',label='NMOR',c=color2)
ax4.set_ylabel('delta B (Hz)', color = color2) 


lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
ax4.legend(lines + lines2, labels + labels2, loc=0)

plt.grid()
plt.show()

vals_FL.sort()
vals_us.sort()

vals_FL = vals_FL/1e-9

coef = np.polyfit(vals_FL,vals_us,1)
poly1d_fn = np.poly1d(coef)

plt.figure()
plt.plot(vals_FL,vals_us,'o',)
plt.ylabel('NMOR Gradiometer measured field (Hz)')
plt.xlabel('FL measured field (nT)')
plt.plot(vals_FL, poly1d_fn(vals_FL), '--k')

plt.grid()
plt.show()


#%% GAIN VALUE

conv = coef[0]*0.071488

gain = 1/conv

print('Gain is: '+ str(gain))





