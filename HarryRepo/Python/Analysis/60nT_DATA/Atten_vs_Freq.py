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

F_Param = [8,16,24,32,40,48,56,80] #Frequency value of each run, in order
# F_Param = [24,28,32,36,40,48]
# F_Param = [8,80,24,40,48,16,56,32]
csv_sep = ';'

amp = 0.5 #nT  #Amplitude of all runs

looper_F = range(len(F_Param)) #loop variable

#Resonance Data
fig1 = plt.figure('fig1')

daq_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230210_094924_10/mag_freqs_000/'
daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230210_094924_10/grad_freqs_000/'

trace_m, trace_legends_m = HCA.DAQ_read_shift(daq_m,csv_sep) #Read in relevant CSVs as Dataframes
trace_g, trace_legends_g = HCA.DAQ_read_shift(daq_g,csv_sep)

traces_m = HCA.DAQ_Tracking_PURE(trace_m,trace_legends_m) #create object
traces_g = HCA.DAQ_Tracking_PURE(trace_g,trace_legends_g)

fs = 837.1

# #sample points
# N_m = traces_m.chunked_time.shape[1]
# N_g = traces_g.chunked_time.shape[1]

# #Spacing

# T_m = traces_m.chunked_time[0,1]-traces_m.chunked_time[0,0]
# T_g = traces_g.chunked_time[0,1]-traces_g.chunked_time[0,0]

# xf_m = fftfreq(N_m, T_m)[:N_m//2]
# xf_g = fftfreq(N_g, T_g)[:N_g//2]

yf_corr_m = np.zeros((len(F_Param),419))
yf_corr_g= np.zeros((len(F_Param),419))

plt.figure()
for a in looper_F:
    # plt.figure()
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
plt.title('Mag vs grad spectra at different frequencies')
plt.xlim(-10,100)
# plt.axvline(x=F_Param[a],ls = '--',c='k',lw = '0.2',label = '{}Hz reference'.format(F_Param[a]))
# plt.ylim(-200,10)
# plt.ylabel('Normalised power (dB)')
plt.xlabel('Frequency')
plt.ylabel('Absolute power (dB)')
plt.grid()
plt.legend()


#Attenuation at frequencies
def freq_peaks_n(freq,frq_domain,spectrum,param,looper):
    
    nidx = np.zeros(len(param))
    maxval = np.zeros(len(param))

    for a in looper:

            nidx[a] = (np.abs(frq_domain-freq[a])).argmin()
            
            roi = spectrum[a,int(nidx[a]-1):int(nidx[a]+1)]
            
            maxval[a] = np.max(roi)
                
    return maxval

peaks_m = freq_peaks_n(F_Param,xf_m,yf_corr_m,F_Param,looper_F)
peaks_g = freq_peaks_n(F_Param,xf_g,yf_corr_g,F_Param,looper_F)

atten = peaks_m-peaks_g

plt.figure()
plt.plot(F_Param,atten,'b*')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation (dB)')
plt.title('Grad vs Mag atteunuation of {}nT sine at 0nT background'.format(amp))
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlim(0,90)
# plt.ylim(20, 35)



    