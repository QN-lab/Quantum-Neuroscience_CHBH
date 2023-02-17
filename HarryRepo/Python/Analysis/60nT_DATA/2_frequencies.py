# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:57:22 2023

@author: H

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
import math
import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal

F_Param = [8,16]
csv_sep = ';'

amp = 0.5 #nT  #Amplitude of all runs

looper_F = range(len(F_Param)) #loop variable

#Resonance Data
fig1 = plt.figure('fig1')

daq_1 = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230213_234331_02/C1_2freq_1_000/'
daq_2 = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230213_234331_02/C2_2freq_1_000/'
daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230213_234331_02/grad_2freq_1_000/'

trace_1, trace_legends_1 = HCA.DAQ_read_shift(daq_1,csv_sep) #Read in relevant CSVs as Dataframes
trace_2, trace_legends_2 = HCA.DAQ_read_shift(daq_2,csv_sep) 
trace_g, trace_legends_g = HCA.DAQ_read_shift(daq_g,csv_sep)

traces_1 = HCA.DAQ_Tracking_PURE(trace_1,trace_legends_1) #create object
traces_2 = HCA.DAQ_Tracking_PURE(trace_2,trace_legends_2)
traces_g = HCA.DAQ_Tracking_PURE(trace_g,trace_legends_g)

fs = 837.1

yf_corr_1 = np.zeros((len(traces_1.patch),419))
yf_corr_2 = np.zeros((len(traces_2.patch),419))
yf_corr_g= np.zeros((len(traces_g.patch),419))

# plt.figure()
for i in traces_1.patch:
    plt.figure()
    xf_1,yf_1 = signal.welch((traces_1.chunked_data[i,:]*0.071488e-9)/3,fs,nperseg = fs,scaling = 'spectrum')
    xf_2,yf_2 = signal.welch((traces_2.chunked_data[i,:]*0.071488e-9)/-1,fs,nperseg = fs,scaling = 'spectrum')
    xf_g,yf_g = signal.welch((traces_g.chunked_data[i,:]*0.071488e-9)/4,fs,nperseg = fs,scaling = 'spectrum')

    yf_corr_1[i,:] = 10*np.log10(yf_1) #dB!
    yf_corr_2[i,:] = 10*np.log10(yf_2) #dB!
    yf_corr_g[i,:] = 10*np.log10(yf_g)
    
    # yf_corr_1[a,:] = yf_1
    # yf_corr_2[a,:] = yf_2
    # yf_corr_g[a,:] = yf_g

    # plt.plot(xf_1, yf_corr_1[i,:],label = 'C1 magnetometer')
    # plt.plot(xf_2, yf_corr_2[i,:],label = 'C2 magnetometer') 
    # plt.plot(xf_g, yf_corr_g[i,:],label = 'gradiometer')
    
    # plt.xlim(-10,100)
    # # plt.axvline(x=F_Param[a],ls = '--',c='k',lw = '0.2',label = '{}Hz reference'.format(F_Param[a]))
    # # plt.ylim(-200,10)
    # plt.ylabel('Power (dB)')
    # plt.xlabel('Frequency')
    # plt.grid()
    # plt.legend()
    
#averaged

y_avg_1 = np.mean(yf_corr_1,axis=0)
y_avg_2 = np.mean(yf_corr_2,axis=0)
y_avg_g = np.mean(yf_corr_g,axis=0)

plt.figure()
plt.plot(xf_1, y_avg_1,label = 'C1 magnetometer')
plt.plot(xf_2, y_avg_2,label = 'C1 magnetometer')
plt.plot(xf_g, y_avg_g,label = 'gradiometer')

plt.xlim(-10,100)
plt.ylabel('Power (dB)')
plt.xlabel('Frequency')
plt.grid()
plt.legend()


    
def freq_peaks_n(freq,frq_domain,spectrum,looper):
    
    nidx = np.zeros(len(freq))
    maxval = np.zeros(len(freq))

    for a in looper:

            nidx[a] = (np.abs(frq_domain-freq[a])).argmin()
            
            roi = spectrum[int(nidx[a]-2):int(nidx[a]+2)]
            
            maxval[a] = np.max(roi)
                
    return maxval

C1_maxs = freq_peaks_n(F_Param,xf_1,y_avg_1,looper_F)

C2_maxs = freq_peaks_n(F_Param,xf_2,y_avg_2,looper_F)

g_maxs = freq_peaks_n(F_Param,xf_g,y_avg_g,looper_F)

gain_C1 = g_maxs-C1_maxs
gain_C2 = g_maxs-C2_maxs

plt.figure()
plt.plot(F_Param,gain_C1,'o',label = 'Grad gain v C1')
plt.plot(F_Param,gain_C2,'o',label = 'Grad gain v C2')
plt.legend()
plt.xlim(0,20)
plt.ylim(-60,20)
plt.grid()





