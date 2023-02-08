# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:32:38 2023

@author: H

Writing the code setup for comparing parameters
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import math
import Harry_analysis as HCA

#%% preamble & User inputs
csv_sep = ';' #separator for saved CSV
sfreq = 2000 #frequency at which to start the fitting
X_Parameter = []
Param_name = 'Heating'

#%% Data Read-In

#Resonance Data
Folderpath_r = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230116_161128_09/Powers_000/' #Resonance Data
res, res_legends = HCA.Res_read(Folderpath_r, csv_sep)

#DAQ Data
Folderpath_daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230110_232908_05/C1_daq_1_000/'

trig, trig_legends = HCA.DAQ_trigger_read(Folderpath_daq,csv_sep)
track, track_legends = HCA.DAQ_tracking_read(Folderpath_daq,csv_sep)
spect, spect_legends = HCA.DAQ_spect_read(Folderpath_daq,csv_sep)

#%%Assign to classes from Harry_analysis module
resonance1 = HCA.Resonance(res, res_legends,sfreq)

trigger1 = HCA.DAQ_Trigger(trig,trig_legends)
tracking1 = HCA.DAQ_Tracking(track,track_legends,trigger1)

spectrum1 = HCA.DAQ_Spectrum(spect,spect_legends,trigger1)

fig1 = plt.figure('fig1')
resonance1.plot_with_fit()

#%%plot ROI on spectrum
s_ind = 511 #Corresponds to 0Hz
f_ind = 634 #"100Hz

fig2 = plt.figure('fig2')
plt.semilogy(spectrum1.frq_domain[s_ind:f_ind],spectrum1.avg_spectrum[s_ind:f_ind])
plt.xlabel("frequency, Hz")
plt.ylabel("quadrature, mV")
plt.grid(color='k', linestyle='-', linewidth=0.5)

fig3 = plt.figure('fig3')
plt.semilogy(spectrum1.frq_domain,spectrum1.avg_spectrum)
plt.semilogy(spectrum1.frq_domain,spectrum1.avg_floor_repd)
plt.xlabel("frequency, Hz")
plt.ylabel("quadrature, mV")
plt.grid(color='k', linestyle='-', linewidth=0.5)

#%%Calculate Sensitivity 
g=1/2
hbar=1.05e-34
mu=9.27e-24
sensitivity=(2*math.pi*resonance1.width*hbar)/(g*mu*spectrum1.avg_SNr)
sens_er = (2*math.pi*resonance1.width_err*hbar)/(g*mu*spectrum1.avg_SNr) #assumes no error on SNr, probably need to account for this
#%% Plotting
fig4= plt.figure('fig4')

#Parameter vs Width
plt.subplot(2,1,1)
plt.errorbar(X_Parameter,resonance1.width,yerr=resonance1.width_err,fmt='.b')
plt.ylim(HCA.limitise(resonance1.width,0.1))
plt.ylabel('Width (Hz)')
plt.grid(color='k', linestyle='-', linewidth=0.5)

#Parameter vs Amplitude
plt.subplot(2,1,2)
plt.errorbar(X_Parameter,resonance1.amplitude,yerr=resonance1.amplitude_err,fmt='.b')
plt.ylim(HCA.limitise(resonance1.amplitude,0.1))
plt.ylabel('Fit Amplitude (mV)')
plt.grid(color='k', linestyle='-', linewidth=0.5)

#Slope of resonance height/width
Fig5 = plt.figure('fig5')
plt.errorbar(X_Parameter,resonance1.h_over_w,yerr=resonance1.h_over_w_err,fmt='.b')
plt.ylim(HCA.limitise(resonance1.h_over_w,0.1))
plt.ylabel('Amplitude over width (mV/Hz)')
plt.xlabel(Param_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)

#Sensitivity
Fig6 = plt.figure('fig6')
plt.errorbar(X_Parameter,sensitivity,yerr=sens_er,fmt='.b')
plt.ylim(HCA.limitise(sensitivity,0.1))
plt.ylabel('Sensitivity (T/rHz)')
plt.xlabel(Param_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)





















