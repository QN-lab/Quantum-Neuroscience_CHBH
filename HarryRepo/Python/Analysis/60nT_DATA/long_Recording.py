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

# F_Param = [8,16,24,32,40,48,56,80]
# F_Param = [24,28,32,36,40,48]

csv_sep = ';'
sfreq = 800

amp = 0.5 #nT 

#Resonance Data
fig1 = plt.figure('fig1')

daq_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230210_094924_10/mag_8Hz_only_long_000/'
daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230210_094924_10/grad_8Hz_only_long_000/'

trace_m, trace_legends_m = HCA.DAQ_tracking_read(daq_m,csv_sep)
trace_g, trace_legends_g = HCA.DAQ_tracking_read(daq_g,csv_sep)

traces_m = HCA.DAQ_Tracking_PURE(trace_m,trace_legends_m)
traces_g = HCA.DAQ_Tracking_PURE(trace_g,trace_legends_g)

F_Param = traces_m.patch

looper_F = range(len(F_Param))
#sample pointsF_Param
N_m = traces_m.chunked_time.shape[1]
N_g = traces_g.chunked_time.shape[1]

#Spacing

T_m = traces_m.chunked_time[0,1]-traces_m.chunked_time[0,0]
T_g = traces_g.chunked_time[0,1]-traces_g.chunked_time[0,0]

xf_m = fftfreq(N_m, T_m)[:N_m//2]
xf_g = fftfreq(N_g, T_g)[:N_g//2]

yf_corr_m = np.zeros((len(F_Param),len(xf_m)))
yf_corr_g= np.zeros((len(F_Param),len(xf_g)))

plt.figure()
for a in looper_F:
    # plt.figure()
    yf_m = (2/N_g)*fft(traces_m.chunked_data[a,:]*0.071488e-9)
    yf_g = (2/N_m)*fft(traces_g.chunked_data[a,:]*0.071488e-9)
    
    yf_corr_m[a,:] = 20*np.log10(np.abs(yf_m[0:N_m//2])/(np.abs(yf_m[0]))) #dB!
    yf_corr_g[a,:] = 20*np.log10(np.abs(yf_g[0:N_g//2])/(np.abs(yf_g[0])))

    
    plt.plot(xf_m, yf_corr_m[a,:],label = 'magnetometer') # Do we need 2/N when doing it in dB???
    plt.plot(xf_g, yf_corr_g[a,:],label = 'gradiometer')
plt.title('Overlay of spectra of all 60 10s runs')
plt.xlim(-10,100)
# plt.axvline(x=F_Param[a],ls = '--',c='k',lw = '0.2',label = '{}Hz reference'.format(F_Param[a]))
plt.ylim(-200,10)
plt.ylabel('Normalised power (dB)')
plt.xlabel('Frequency')
plt.grid()
# plt.legend()

#Attenuation figures

def freq_peaks_n(freq,frq_domain,spectrum,param,looper):
    
    nidx = np.zeros(len(param))
    maxval = np.zeros(len(param))

    for a in looper:

            nidx[a] = (np.abs(frq_domain-freq)).argmin()
            
            roi = spectrum[a,int(nidx[a]-2):int(nidx[a]+2)]
            
            maxval[a] = np.max(roi)
                
    return maxval

peaks_m = freq_peaks_n(8,xf_m,yf_corr_m,F_Param,looper_F)
peaks_g = freq_peaks_n(8,xf_g,yf_corr_g,F_Param,looper_F)

atten = peaks_m-peaks_g

plt.figure()
plt.plot(F_Param,atten,'b*')
plt.xlabel('Run number')
plt.ylabel('Attenuation (dB)')
plt.title('Long-run Grad vs Mag attenuation of 0.5nT 8Hz sine at 0nT offset')
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlim(-5,65)
# plt.ylim(20, 35)



    