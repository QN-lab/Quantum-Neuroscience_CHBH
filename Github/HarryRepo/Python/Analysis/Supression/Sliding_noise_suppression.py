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

#%% preamble & User inputs
csv_sep = ';' #separator for saved CSV
sfreq = 800 #frequency domain value at which to start the fitting

A_Param = [0.350, 1] #2.5,4.5
A_Param_name = 'Noise Amplitude (nT)'

F_Param = [5,10,15,20,25,30,35,40,45,50,75,100,150,200]
F_Param_name = 'Noise Frequency (nT)'

A_ind = 1

#%% Data Read-In

looper_A = range(len(A_Param))
looper_F = range(len(F_Param))

resonance_m = list([])
resonance_g = list([])
#Resonance Data
fig1 = plt.figure('fig1')
for a in looper_A:
    
    Folderpath_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230131_121127_08/mag_{}nT_res_000/'.format(A_Param[a]) #Resonance Data
    res_m, res_legends_m = HCA.Res_read(Folderpath_m, csv_sep)
    
    resonance_m.append(HCA.Resonance(res_m, res_legends_m, sfreq))
    # resonance_m[a].plot_with_fit()
    

#DAQ Data
spectrum_m = list([])
spectrum_g = list([])


for a in looper_A:
    daq_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230131_121127_08/mag_{}nT_noise_000/'.format(A_Param[a])
    daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230131_121127_08/grad_{}nT_noise_000/'.format(A_Param[a])
    
    spect_m, spect_legends_m = HCA.Spect_read(daq_m,csv_sep)
    spect_g, spect_legends_g = HCA.Spect_read(daq_g,csv_sep)
    
    spectrum_m.append(HCA.Spect_analyser(spect_m,spect_legends_m))
    spectrum_g.append(HCA.Spect_analyser(spect_g,spect_legends_g))

#%% Spectra (Tempwise)

for f in looper_F:
    fig= plt.figure('fig' + str(f+1))
    
    plt.plot(spectrum_m[A_ind].frq_domain,spectrum_m[A_ind].chunked_data[f],label='Magnetometer')
    plt.plot(spectrum_g[A_ind].frq_domain,spectrum_g[A_ind].chunked_data[f],label='Gradiometer')
    
    # plt.title('FREQUENCY = {}'.format(F_param))
    plt.xlabel("frequency, Hz")
    plt.ylabel("Power (dBV^2/Hz)")
    plt.ylim(HCA.limitise(spectrum_g[A_ind].chunked_data[f],0.2))
    plt.title('FREQ = {} Hz'.format(F_Param[f]))
    plt.legend()
    plt.grid(color='k', linestyle='-', linewidth=0.5)

#ATTENUATION

max_m1 = spectrum_m[0].extract_peaks()
max_g1 = spectrum_g[0].extract_peaks()

max_m2 = spectrum_m[1].extract_peaks()
max_g2 = spectrum_g[1].extract_peaks()

max_m = [max_m1, max_m2]
max_g = [max_g1, max_g2]

atten = np.zeros((len(A_Param),len(max_m[0][0])))

for a in looper_A:
    atten[a,:] = max_m[a][0]-max_g[a][0]

fig1000 = plt.figure('fig1000')
for a in looper_A:
    plt.plot(max_m[1][1],atten[a,:],'.',label = str(A_Param[a]))
plt.title('Noise attenuation of gradiometer vs magnetometer')

plt.xlabel(F_Param_name)
plt.ylabel('Attenuation (dB)')
plt.legend(title= 'Noise Amp (nT)',loc='lower right')
plt.xlim(-10,220)
plt.grid(color='k', linestyle='-', linewidth=0.5)








