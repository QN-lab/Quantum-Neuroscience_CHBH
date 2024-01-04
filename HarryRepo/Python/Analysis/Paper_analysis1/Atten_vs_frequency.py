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
import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd

A_Param = [8,14,20,26,32,38,44,50,56,62,68,74,80]
A_Param_r = A_Param[::-1]
csv_sep = ';'

updown = ['asc','des'] # 0nT asc,des; 60nT asc,des
locks = [0,60]

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

        daq_m = 'Z:/Data/2023_02_20/{}nT/mag_freqs_{}_000/'.format(j,i)
        daq_g = 'Z:/Data/2023_02_20/{}nT/grad_freqs_{}_000/'.format(j,i)
        
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
        # yf_chunked[a,:] = 10*np.log10(yf) #dB!
        yf_chunked[a,:] = yf #POWER!!!!
    
    return xf, yf_chunked

# traces_m[0].time_domain, traces_m[0].power = Powerise(A_Param,traces_m[0].chunked_data,looper_A)
# traces_m[1].time_domain, traces_m[1].power = Powerise(A_Param_r,traces_m[1].chunked_data,looper_A)

# traces_m[2].time_domain, traces_m[2].power = Powerise(A_Param,traces_m[2].chunked_data,looper_A)
# traces_m[3].time_domain, traces_m[3].power = Powerise(A_Param_r,traces_m[3].chunked_data,looper_A)


# traces_g[0].time_domain, traces_g[0].power = Powerise(A_Param,traces_g[0].chunked_data,looper_A)
# traces_g[1].time_domain, traces_g[1].power = Powerise(A_Param_r,traces_g[1].chunked_data,looper_A)

# traces_g[2].time_domain, traces_g[2].power = Powerise(A_Param,traces_g[2].chunked_data,looper_A)
# traces_g[3].time_domain, traces_g[3].power = Powerise(A_Param_r,traces_g[3].chunked_data,looper_A)

def Powerise_fft(field_data,chunk):
    
    N = field_data.shape[1]
    T = 0.0011946666666666666 #chunked_time[0,1]-chunked_time[0,0]
    xf = fftfreq(N,T)[:N//2]
    
    yf_chunked = np.zeros((len(chunk),len(xf)))

    for a in chunk:
        
        yf = (2/N)*fft(field_data[a,:])
    
        yf_chunked[a,:] = np.abs(yf[0:N//2]) # AMPLITUDE!!! NOT POWER
    
    return xf, yf_chunked

traces_m[0].time_domain, traces_m[0].power = Powerise_fft(traces_m[0].chunked_field,looper_A)
traces_m[1].time_domain, traces_m[1].power = Powerise_fft(traces_m[1].chunked_field,looper_A)

traces_m[2].time_domain, traces_m[2].power = Powerise_fft(traces_m[2].chunked_field,looper_A)
traces_m[3].time_domain, traces_m[3].power = Powerise_fft(traces_m[3].chunked_field,looper_A)


traces_g[0].time_domain, traces_g[0].power = Powerise_fft(traces_g[0].chunked_field,looper_A)
traces_g[1].time_domain, traces_g[1].power = Powerise_fft(traces_g[1].chunked_field,looper_A)

traces_g[2].time_domain, traces_g[2].power = Powerise_fft(traces_g[2].chunked_field,looper_A)
traces_g[3].time_domain, traces_g[3].power = Powerise_fft(traces_g[3].chunked_field,looper_A)

def freq_peaks_n(freq,frq_domain,spectrum,param,looper):
    
    nidx = np.zeros(len(param))
    maxval = np.zeros(len(param))

    for a in looper:

            nidx[a] = (np.abs(frq_domain-freq[a])).argmin()
            
            roi = spectrum[a,int(nidx[a]-1):int(nidx[a]+1)]
            
            maxval[a] = np.max(roi)
                
    return maxval

peaks_m = np.zeros((4,13))
peaks_g = np.zeros((4,13))
for i in [0,2]:
        peaks_m[i,:] = freq_peaks_n(A_Param,traces_m[i].time_domain,traces_m[i].power,A_Param,looper_A) 
        peaks_g[i,:] = freq_peaks_n(A_Param,traces_g[i].time_domain,traces_g[i].power,A_Param,looper_A) 
        
for i in [1,3]:
        peaks_m[i,:] = freq_peaks_n(A_Param_r,traces_m[i].time_domain,traces_m[i].power,A_Param,looper_A) 
        peaks_g[i,:] = freq_peaks_n(A_Param_r,traces_g[i].time_domain,traces_g[i].power,A_Param,looper_A)        

# atten = np.subtract(peaks_m,peaks_g) #do on 'dB' peaks
atten = np.divide(peaks_m,peaks_g)    #do on 'factor' peaks

#Remove 50Hz points########################
l0 = atten[0,np.arange(len(atten[0,:]))!=7]
l1 = atten[1,np.arange(len(atten[1,:]))!=5]
l2 = atten[2,np.arange(len(atten[2,:]))!=7]
l3 = atten[3,np.arange(len(atten[3,:]))!=5]

atten_f = np.vstack([l0,l1,l2,l3])

A_Param = np.delete(A_Param,[7],0)
A_Param_r = np.delete(A_Param_r,[5],0)
###########################################

plt.figure()
plt.plot(A_Param,atten_f[0,:],'-o',label = '0nT; asc')
plt.plot(A_Param_r,atten_f[1,:],'-o',label = '0nT; des')
plt.plot(A_Param,atten_f[2,:],'-o',label = '60nT; asc')
plt.plot(A_Param_r,atten_f[3,:],'-o',label = '60nT; des')
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlabel('Applied frequency of "noise" (Hz)')
plt.ylabel('Attenuation (factor)')
plt.legend(title = 'Bo; run type',loc = 'upper right',fontsize=10)
plt.xlim(0,85)
plt.title('Attenuation of 0.5nT Sine at varying frequencies')
plt.show()

#%% AVG with Errors

atten_0 = np.vstack([atten_f[0,:],atten_f[1,::-1]])
atten_60 = np.vstack([atten_f[2,:],atten_f[3,::-1]])

avg_0 = np.mean(atten_0,axis=0)
avg_60 = np.mean(atten_60,axis=0)

r_0 = abs(atten_0-avg_0).max(axis=0)
r_60 = abs(atten_60-avg_60).max(axis=0)

err_0 = np.mean(r_0)
err_60 = np.mean(r_60)

plt.figure()

plt.errorbar(A_Param,avg_0,yerr = err_0,fmt='-o',label = '0nT')
plt.errorbar(A_Param,avg_60,yerr = err_60,fmt='-o',label = '60nT')
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlabel('Applied frequency of "noise" (Hz)')
plt.ylabel('Attenuation (factor)')
plt.legend(title = 'Bo; run type',loc = 'upper right',fontsize=10)
plt.xlim(0,85)
plt.title('Attenuation of 0.5nT Sine at varying frequencies')
plt.show()






