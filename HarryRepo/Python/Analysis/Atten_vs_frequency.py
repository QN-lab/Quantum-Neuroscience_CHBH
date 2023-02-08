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

F_Param = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
csv_sep = ';'
sfreq = 800

amp = 0.5 # Frequency of interest

looper_F = range(len(F_Param))

#Resonance Data
fig1 = plt.figure('fig1')

Folderpath_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230206_102048_00/mag_{}nT_res_000/'.format(amp)
res_m, res_legends_m = HCA.Res_read(Folderpath_m, csv_sep)

resonance_m = HCA.Resonance(res_m, res_legends_m, sfreq)

Folderpath_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230206_102048_00/grad_{}nT_res_000/'.format(amp)
res_g, res_legends_g = HCA.Res_read(Folderpath_g, csv_sep)

resonance_g = HCA.Resonance(res_g, res_legends_g, sfreq)

# resonance_m.plot_with_fit()
# resonance_g.plot_with_fit()

#DAQ Data
# spectrum_m = list([])
# spectrum_g = list([])


daq_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230206_102048_00/mag_{}nT_noise_000/'.format(amp)
daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230206_102048_00/grad_{}nT_noise_000/'.format(amp)

trace_m, trace_legends_m = HCA.DAQ_read_shift(daq_m,csv_sep)
trace_g, trace_legends_g = HCA.DAQ_read_shift(daq_g,csv_sep)

traces_m = HCA.DAQ_Tracking_PURE(trace_m,trace_legends_m)
traces_g = HCA.DAQ_Tracking_PURE(trace_g,trace_legends_g)

#sample points
N_m = traces_m.chunked_time.shape[1]
N_g = traces_g.chunked_time.shape[1]

#Spacing

T_m = traces_m.chunked_time[0,1]-traces_m.chunked_time[0,0]
T_g = traces_g.chunked_time[0,1]-traces_g.chunked_time[0,0]
        


xf_m = fftfreq(N_m, T_m)[:N_m//2]
xf_g = fftfreq(N_g, T_g)[:N_g//2]

yf_corr_m = np.zeros((len(F_Param),len(xf_m)))
yf_corr_g= np.zeros((len(F_Param),len(xf_g)))

for a in looper_F:
    
    yf_m = fft(traces_m.chunked_data[a,:]*0.071488e-9)
    yf_g = fft(traces_g.chunked_data[a,:]*0.071488e-9)
    
    yf_corr_m[a,:] = 20*np.log10(np.abs(yf_m[0:N_m//2])) #dB!
    yf_corr_g[a,:] = 20*np.log10(np.abs(yf_g[0:N_g//2]))

    plt.figure()
    plt.plot(xf_m, 2/N_m * yf_corr_m[a,:],label = 'magnetometer') # Do we need 2/N when doing it in dB???
    plt.plot(xf_g, 2/N_g * yf_corr_g[a,:],label = 'gradiometer')
    plt.title('Frequency: {}Hz'.format(F_Param[a]))
    plt.xlim(-10,150)
    # plt.axvline(x=freq,ls = '--',c='k',lw = '0.2',label = '{}Hz reference'.format(freq))
    plt.grid()
    plt.legend()

#Attenuation figures

def freq_peaks_n(freq,frq_domain,spectrum,param,looper):
    
    nidx = np.zeros(len(param))
    maxval = np.zeros(len(param))

    for a in looper:

            nidx[a] = (np.abs(frq_domain-freq[a])).argmin()
            
            roi = spectrum[a,int(nidx[a]-2):int(nidx[a]+2)]
            
            maxval[a] = np.max(roi)
                
    return maxval

peaks_m = freq_peaks_n(F_Param,xf_m,yf_corr_m,F_Param,looper_F)
peaks_g = freq_peaks_n(F_Param,xf_g,yf_corr_g,F_Param,looper_F)

atten = peaks_m-peaks_g

plt.figure()
plt.plot(F_Param,atten,'b*')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation (dB)')
plt.title('Grad vs Mag attenuation @ {}nT'.format(amp))
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlim(0,105)
plt.ylim(20, 35)



    