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

# F_Param = [8,16,24,32,40,48,56,80]
# F_Param = [24,28,32,36,40,48]
F_Param = [0.5,0.75,1,2,3,5]
csv_sep = ';'
sfreq = 800
freq = 80

looper_F = range(len(F_Param))

#Resonance Data
fig1 = plt.figure('fig1')

daq_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230215_152637_00/mag_80Hz_60lock_000/'
daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230215_152637_00/grad_80Hz_60lock_000/'

trace_m, trace_legends_m = HCA.DAQ_tracking_read(daq_m,csv_sep)
trace_g, trace_legends_g = HCA.DAQ_tracking_read(daq_g,csv_sep)

traces_m = HCA.DAQ_Tracking_PURE(trace_m,trace_legends_m)
traces_g = HCA.DAQ_Tracking_PURE(trace_g,trace_legends_g)

fs = 837.1

yf_corr_m = np.zeros((len(F_Param),419))
yf_corr_g= np.zeros((len(F_Param),419))

# plt.figure()
for a in looper_F:
    plt.figure()
    xf_m,yf_m = signal.welch(traces_m.chunked_data[a,:]*0.071488e-9,fs,nperseg = fs)
    xf_g,yf_g = signal.welch(traces_g.chunked_data[a,:]*0.071488e-9,fs,nperseg = fs)
    
    yf_corr_m[a,:] = 10*np.log10(yf_m) #dB!
    yf_corr_g[a,:] = 10*np.log10(yf_g)
    # yf_corr_m[a,:] = yf_m
    # yf_corr_g[a,:] = yf_g
    
    # plt.plot(xf_m, yf_corr_m[a,:],label = 'magnetometer') 
    # plt.plot(xf_g, yf_corr_g[a,:],label = 'gradiometer')
    
    if a == 0:
        plt.plot(xf_m, yf_corr_m[a,:],'b',label = 'magnetometer') 
        plt.plot(xf_g, yf_corr_g[a,:],'g',label = 'gradiometer')
    elif a > 0:
        plt.plot(xf_m, yf_corr_m[a,:],'b') 
        plt.plot(xf_g, yf_corr_g[a,:],'g')
    plt.xlim(-10,100)
    # plt.ylim(-200,10)
    plt.ylabel('Normalised power (dB)')
    plt.xlabel('Frequency')
    plt.grid()
    plt.legend()

#Attenuation figures

def freq_peaks(freq,frq_domain,spectrum,param,looper):
    
    nidx = np.zeros(len(param))
    maxval = np.zeros(len(param))

    for a in looper:

            nidx[a] = (np.abs(frq_domain-freq)).argmin()
            
            roi = spectrum[a,int(nidx[a]-2):int(nidx[a]+2)]
            
            maxval[a] = np.max(roi)
                
    return maxval

peaks_m = freq_peaks(freq,xf_m,yf_corr_m,F_Param,looper_F)
peaks_g = freq_peaks(freq,xf_g,yf_corr_g,F_Param,looper_F)

atten = peaks_m-peaks_g

plt.figure()
plt.plot(F_Param,atten,'b*')
plt.xlabel('Amplitude of applied HG field (nT)')
plt.ylabel('Attenuation (dB)')
plt.title('Grad vs Mag at 0nT lock with atteunuation of 8Hz sine at different HG fields')
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlim(0,6)
# plt.ylim(20, 35)



    