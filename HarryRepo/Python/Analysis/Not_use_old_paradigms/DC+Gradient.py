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

F_Param = [10,25,50,100,150,200,250] #Added gradient
csv_sep = ';'
sfreq = 800

amp = 2 #Amplitude of DC offset

looper_F = range(len(F_Param))


daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230206_171520_02/{}nT_DC_noise_000/'.format(amp)

trace_g, trace_legends_g = HCA.DAQ_read_shift(daq_g,csv_sep)

traces_g = HCA.DAQ_Tracking_PURE(trace_g,trace_legends_g)


N_g = traces_g.chunked_time.shape[1]

T_g = traces_g.chunked_time[0,1]-traces_g.chunked_time[0,0]

xf_g = fftfreq(N_g, T_g)[:N_g//2]

yf_corr_g= np.zeros((len(F_Param),len(xf_g)))

for a in looper_F:

    yf_g = fft(traces_g.chunked_data[a,:]*0.071488e-9)
    
    yf_corr_g[a,:] = np.abs(yf_g[0:N_g//2])

    plt.figure()
    plt.semilogy(xf_g, 2/N_g * yf_corr_g[a,:])
    plt.title('Frequency: {}Hz'.format(F_Param[a]))
    plt.xlim(-10,150)
    # plt.axvline(x=freq,ls = '--',c='k',lw = '0.2',label = '{}Hz reference'.format(freq))
    plt.grid()
    # plt.legend()

#Attenuation figures

def freq_peaks_n(freq,frq_domain,spectrum,param,looper):
    
    nidx = np.zeros(len(param))
    maxval = np.zeros(len(param))

    for a in looper:

            nidx[a] = (np.abs(frq_domain-freq[a])).argmin()
            
            roi = spectrum[a,int(nidx[a]-2):int(nidx[a]+2)]
            
            maxval[a] = np.max(roi)
                
    return maxval

peaks_g = freq_peaks_n(F_Param,xf_g,yf_corr_g,F_Param,looper_F)

plt.figure()
plt.plot(F_Param,peaks_g/1e-12,'b*')
plt.xlabel('added gradient(pT)')
plt.ylabel('measured gradient(pT)')
plt.title('applied v measured gradient at {}nT DC offset'.format(amp))
plt.grid(color='k', linestyle='-', linewidth=0.3)
# plt.xlim(0,105)
# plt.ylim(20, 35)



    