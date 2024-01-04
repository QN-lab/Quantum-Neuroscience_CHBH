# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:08:32 2023

@author: H
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('classic')
import math
# import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd

A_Param = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
A_Param_r = [5,4.5,4,3.5,3,2.5,2,1.5,1,0.5]
csv_sep = ';'

updown = ['asc','des'] # 0nT asc,des; 60nT asc,des
locks = [0,60]

freq = 80 # Frequency of interest
looper_A = range(len(A_Param))

traces_m = list()
traces_g = list()

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
#%% Load in objects in order
    #0nT: up,down; 60nT up,down
for j in locks:
     for i in updown:

        daq_m = 'Z:/Data/2023_02_20/{}nT/mag_80Hz_{}_000/'.format(j,i)
        daq_g = 'Z:/Data/2023_02_20/{}nT/grad_80Hz_{}_000/'.format(j,i)
        
        trace_m, trace_legends_m = DAQ_read_shift(daq_m,csv_sep)
        trace_g, trace_legends_g = DAQ_read_shift(daq_g,csv_sep)
    
        traces_m.append(DAQ_Tracking_PURE(trace_m,trace_legends_m))
        traces_g.append(DAQ_Tracking_PURE(trace_g,trace_legends_g))
    

#%% Add spectra and spectral domain to each object
def Powerise(A_Param,chunked_data,looper_A):
    
    
    fs = 837.1
    
    xf_chunked = np.zeros((len(A_Param),419))
    yf_chunked = np.zeros((len(A_Param),419))

    for a in looper_A:
    
        xf,yf = signal.welch(chunked_data[a,:]*0.071488e-9,fs,nperseg = fs)
    
        xf_chunked[a,:] = xf
        yf_chunked[a,:] = 10*np.log10(yf) #dB!
    
    return xf, yf_chunked

traces_m[0].time_domain, traces_m[0].power = Powerise(A_Param,traces_m[0].chunked_data,looper_A)
traces_m[1].time_domain, traces_m[1].power = Powerise(A_Param_r,traces_m[1].chunked_data,looper_A)

traces_m[2].time_domain, traces_m[2].power = Powerise(A_Param,traces_m[2].chunked_data,looper_A)
traces_m[3].time_domain, traces_m[3].power = Powerise(A_Param_r,traces_m[3].chunked_data,looper_A)


traces_g[0].time_domain, traces_g[0].power = Powerise(A_Param,traces_g[0].chunked_data,looper_A)
traces_g[1].time_domain, traces_g[1].power = Powerise(A_Param_r,traces_g[1].chunked_data,looper_A)

traces_g[2].time_domain, traces_g[2].power = Powerise(A_Param,traces_g[2].chunked_data,looper_A)
traces_g[3].time_domain, traces_g[3].power = Powerise(A_Param_r,traces_g[3].chunked_data,looper_A)


def freq_peaks(freq,frq_domain,spectrum,param,looper):
    
    nidx = np.zeros(len(param))
    maxval = np.zeros(len(param))

    for a in looper:

            nidx[a] = (np.abs(frq_domain-freq)).argmin()
            
            roi = spectrum[a,int(nidx[a]-2):int(nidx[a]+2)]
            
            maxval[a] = np.max(roi)
                
    return maxval

peaks_m = np.zeros((4,10))
peaks_g = np.zeros((4,10))
for i in [0,1,2,3]:
        peaks_m[i,:] = freq_peaks(freq,traces_m[i].time_domain,traces_m[i].power,A_Param,looper_A) 
        peaks_g[i,:] = freq_peaks(freq,traces_g[i].time_domain,traces_g[i].power,A_Param,looper_A) 

atten1 = np.subtract(peaks_m,peaks_g)

atten1_0 = np.vstack([atten1[0,:],atten1[1,::-1]])
atten1_60 = np.vstack([atten1[2,:],atten1[3,::-1]])


avg1_0 = np.mean(atten1_0,axis=0)
avg1_60 = np.mean(atten1_60,axis=0)

r1_0 = abs(atten1_0-avg1_0).max(axis=0)
r1_60 = abs(atten1_60-avg1_60).max(axis=0)

err1_0 = np.mean(r1_0)
err1_60 = np.mean(r1_60)


plt.figure()
plt.errorbar(A_Param,avg1_0,yerr = err1_0,fmt='-o',label = '0nT')
plt.errorbar(A_Param,avg1_60,yerr = err1_60,fmt='-o',label = '60nT')
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlabel('Applied sinusoidal "noise", b (nT)')
plt.ylabel('Attenuation (dB)')
plt.legend(title = 'Bo',loc = 'lower left',fontsize=10)
plt.xlim(-0.5,5.5)
plt.title('Attenuation of {}Hz applied noise, b'.format(freq))
plt.show()

#%% 8Hz (exact copy except one less value)

A_Param = [0.5,1,1.5,2,2.5,3,3.5,4,4.5]
A_Param_r = [4.5,4,3.5,3,2.5,2,1.5,1,0.5]

traces_m = list()
traces_g = list()
freq = 8
looper_A = range(len(A_Param))

for j in locks:
     for i in updown:

        daq_m = 'Z:/Data/2023_02_20/{}nT/mag_8Hz_{}_000/'.format(j,i)
        daq_g = 'Z:/Data/2023_02_20/{}nT/grad_8Hz_{}_000/'.format(j,i)
        
        trace_m, trace_legends_m = DAQ_read_shift(daq_m,csv_sep)
        trace_g, trace_legends_g = DAQ_read_shift(daq_g,csv_sep)
    
        traces_m.append(DAQ_Tracking_PURE(trace_m,trace_legends_m))
        traces_g.append(DAQ_Tracking_PURE(trace_g,trace_legends_g))

traces_m[0].time_domain, traces_m[0].power = Powerise(A_Param,traces_m[0].chunked_data,looper_A) #there is no final run
traces_m[1].time_domain, traces_m[1].power = Powerise(A_Param_r,traces_m[1].chunked_data,looper_A)

traces_m[2].time_domain, traces_m[2].power = Powerise(A_Param,traces_m[2].chunked_data,looper_A)
traces_m[3].time_domain, traces_m[3].power = Powerise(A_Param_r,traces_m[3].chunked_data,looper_A)


traces_g[0].time_domain, traces_g[0].power = Powerise(A_Param,traces_g[0].chunked_data,looper_A)
traces_g[1].time_domain, traces_g[1].power = Powerise(A_Param_r,traces_g[1].chunked_data,looper_A)

traces_g[2].time_domain, traces_g[2].power = Powerise(A_Param,traces_g[2].chunked_data,looper_A)
traces_g[3].time_domain, traces_g[3].power = Powerise(A_Param_r,traces_g[3].chunked_data,looper_A)


peaks_m = np.zeros((4,9))
peaks_g = np.zeros((4,9))
for i in [0,1,2,3]:
        peaks_m[i,:] = freq_peaks(freq,traces_m[i].time_domain,traces_m[i].power,A_Param,looper_A) 
        peaks_g[i,:] = freq_peaks(freq,traces_g[i].time_domain,traces_g[i].power,A_Param,looper_A) 

atten2 = np.subtract(peaks_m,peaks_g)

atten2_0 = np.vstack([atten2[0,:],atten2[1,::-1]])
atten2_60 = np.vstack([atten2[2,:],atten2[3,::-1]])


avg2_0 = np.mean(atten2_0,axis=0)
avg2_60 = np.mean(atten2_60,axis=0)

r2_0 = abs(atten2_0-avg2_0).max(axis=0)
r2_60 = abs(atten2_60-avg2_60).max(axis=0)

err2_0 = np.mean(r2_0)
err2_60 = np.mean(r2_60)

plt.figure()
plt.errorbar(A_Param,avg2_0,yerr = err2_0,fmt='-o',label = '0nT')
plt.errorbar(A_Param,avg2_60,yerr = err2_60,fmt='-o',label = '60nT')
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlabel('Applied sinusoidal "noise", b (nT)')
plt.ylabel('Attenuation (dB)')
plt.legend(title = 'Bo',loc = 'lower left',fontsize=10)
plt.xlim(-0.5,5.5)
plt.title('Attenuation of {}Hz applied noise, b'.format(freq))
plt.show()
















