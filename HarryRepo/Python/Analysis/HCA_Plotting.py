# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:32:38 2023

@author: H
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import pandas as pd
import math
import Harry_analysis as HCA

#%% preamble & User inputs
csv_sep = ';' #separator for saved CSV
sfreq = 2000 #frequency at which to start the fitting

#%% Data Read-In

def ReadData(folder_path,filename,headername):
    output_sig = pd.read_csv(folder_path+filename,sep=csv_sep)
    output_headers = pd.read_csv(folder_path+headername,sep=csv_sep)
    
    return output_sig, output_headers


#Resonance Data
Folderpath_r = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230112_103211_06/grad_res_000/'     
Filename_r = 'dev3994_demods_0_sample_00000.csv'
Headername_r = 'dev3994_demods_0_sample_header_00000.csv'

res, res_legends = ReadData(Folderpath_r,Filename_r,Headername_r)

#DAQ Data
Folderpath_daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230112_103211_06/grad_DAQ_grad_and_noise_000/'

#DAQ Spectrum
Filename_spectrum = 'dev3994_demods_0_sample_xiy_fft_abs_avg_00000.csv' #Spectrum Data
Headername_s = 'dev3994_demods_0_sample_xiy_fft_abs_avg_header_00000.csv' #Headers

spect, spect_legends = ReadData(Folderpath_daq,Filename_spectrum,Headername_s)

#Signal channel
Filename_signal = 'dev3994_demods_0_sample_frequency_avg_00000.csv'
Headername_signal = 'dev3994_demods_0_sample_frequency_avg_header_00000.csv' 

track, track_legends = ReadData(Folderpath_daq,Filename_signal,Headername_signal)

#Trigger
Filename_trigger = 'dev3994_demods_0_sample_trigin2_avg_00000.csv'
Headername_trigger = 'dev3994_demods_0_sample_trigin2_avg_header_00000.csv'

trig, trig_legends = ReadData(Folderpath_daq,Filename_trigger,Headername_trigger)

#%%ASSIGN TO CLASSES from Harry_analysis module
resonance = HCA.Resonance(res, res_legends,sfreq)

trigger = HCA.DAQ_Trigger(trig,trig_legends)
tracking = HCA.DAQ_Tracking(track,track_legends,trigger)


spectrum = HCA.DAQ_Spectrum(spect,spect_legends,trigger)

fig1 = plt.figure('fig1')
resonance.plot_with_fit()

#%%plot ROI on spectrum
s_ind = 511 #Corresponds to 0Hz
f_ind = 634 #"100Hz

fig2 = plt.figure('fig2')
plt.semilogy(spectrum.frq_domain[s_ind:f_ind],spectrum.avg_spectrum[s_ind:f_ind])
plt.xlabel("frequency, Hz")
plt.ylabel("quadrature, mV")
plt.grid(color='k', linestyle='-', linewidth=0.5)

fig3 = plt.figure('fig3')
plt.semilogy(spectrum.frq_domain,spectrum.avg_spectrum)
plt.semilogy(spectrum.frq_domain,spectrum.avg_floor_repd)
plt.xlabel("frequency, Hz")
plt.ylabel("quadrature, mV")
plt.grid(color='k', linestyle='-', linewidth=0.5)
#%%Calculate Sensitivity 
g=1/2
hbar=1.05e-34
mu=9.27e-24
sensitivity=(2*math.pi*resonance.width*hbar)/(g*mu*spectrum.avg_SNr)

print('Sensitivity in T/rHz: {}'.format(sensitivity))
print('      (Based on noise floor in spectrum figure)')






















