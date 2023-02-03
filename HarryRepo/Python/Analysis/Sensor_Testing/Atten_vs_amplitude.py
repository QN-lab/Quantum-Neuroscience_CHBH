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

A_Param = [0.4,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
csv_sep = ';'
sfreq = 800

looper_A = range(len(A_Param))

#Resonance Data
fig1 = plt.figure('fig1')

Folderpath_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230201_111602_09/mag_8Hz_res_000/'
res_m, res_legends_m = HCA.Res_read(Folderpath_m, csv_sep)

resonance_m = HCA.Resonance(res_m, res_legends_m, sfreq)

Folderpath_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230201_111602_09/grad_8Hz_res_000/'
res_g, res_legends_g = HCA.Res_read(Folderpath_g, csv_sep)

resonance_g = HCA.Resonance(res_g, res_legends_g, sfreq)

# resonance_m.plot_with_fit()
# resonance_g.plot_with_fit()

#DAQ Data
# spectrum_m = list([])
# spectrum_g = list([])


daq_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230201_111602_09/mag_8Hz_noise_000/'
daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230201_111602_09/grad_8Hz_noise_000/'

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

yf_corr_m = np.zeros((len(A_Param),len(xf_m)))
yf_corr_g= np.zeros((len(A_Param),len(xf_g)))

for a in looper_A:
    
    yf_m = fft(traces_m.chunked_data[a,:]*0.07488e-9)
    yf_g = fft(traces_g.chunked_data[a,:]*0.07488e-9)
    
    # yf_corr_m[a,:] = 20*np.log10(np.abs(yf_m[0:N_m//2])) #dB!
    # yf_corr_g[a,:] = 20*np.log10(np.abs(yf_g[0:N_g//2]))

    yf_corr_m[a,:] = np.abs(yf_m[0:N_m//2]) #Amp spectrum
    yf_corr_g[a,:] = np.abs(yf_g[0:N_g//2])
    
    plt.figure()
    plt.semilogy(xf_m, 2/N_m * yf_corr_m[a,:],label = 'magnetometer')
    plt.semilogy(xf_g, 2/N_g * yf_corr_g[a,:],label = 'gradiometer')
    plt.title('Amplitude: {}nT'.format(A_Param[a]))
    plt.xlim(-10,150)
    plt.axvline(x=8,ls = '--',c='k',lw = '0.2')
    plt.grid()
    plt.legend()


freq = 8
#Attenuation

def freq_peaks(freq,frq_domain,spectrum,param,looper):
    
    nidx = np.zeros(len(param))
    maxval = np.zeros(len(param))

    for a in looper:

            nidx[a] = (np.abs(frq_domain-freq)).argmin()
            
            roi = spectrum[a,int(nidx[a]-2):int(nidx[a]+2)]
            
            maxval[a] = np.max(roi)
                
    return maxval

peaks_m = freq_peaks(freq,xf_m,yf_corr_m,A_Param,looper_A)
peaks_g = freq_peaks(freq,xf_m,yf_corr_g,A_Param,looper_A)

atten = peaks_m-peaks_g

plt.figure()
plt.plot(A_Param,atten,'b.')
plt.xlabel('Noise Amplitude (nT)')
plt.ylabel('Attenuation (dB)')

plt.grid(color='k', linestyle='-', linewidth=0.3)



    