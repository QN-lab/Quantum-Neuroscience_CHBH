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
sfreq = 800 #frequency at which to start the fitting
X_Parameter = [0.125,0.150,0.175,0.200,0.225,0.250,0.275,0.300,0.325,0.350]
runs = [0,1,2,3,4,5,6,7,8,9]
Param_name = 'Detuning (V)'

#%% Data Read-In

#Resonance Data
Folderpath_r = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230117_230951_01/resonances_compare_001/' #Resonance Data
res, res_legends = HCA.Res_read(Folderpath_r, csv_sep)

#Create resonance object
resonance = HCA.Resonance(res, res_legends, sfreq)

#Access plotting function from resonance object to fit and plot etc. 
fig1 = plt.figure('fig1')
resonance.plot_no_fit()

#DAQ Data
trigger = list([])
tracking = list([])
spectrum = list([])

daq = list([])
daq.append('C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230117_230951_01/P_box_Bias_000/')
daq.append('C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230117_230951_01/Zurich_Bias_000/')

looper = range(len(daq))
#Create lists of objects, each corresponding to every DAQ run
for i in looper:
    trig, trig_legends = HCA.DAQ_trigger_read(daq[i],csv_sep)
    track, track_legends = HCA.DAQ_tracking_read(daq[i],csv_sep)
    spect, spect_legends = HCA.DAQ_spect_read(daq[i],csv_sep)
    
    #Lists of objects
    trigger.append(HCA.DAQ_Trigger(trig,trig_legends))
    tracking.append(HCA.DAQ_Tracking(track,track_legends,trigger[i]))
    spectrum.append(HCA.DAQ_Spectrum(spect,spect_legends,trigger[i]))
    
#%%plot ROI on spectrum
s_ind = 511 #Corresponds to 0Hz
f_ind = 634 #100Hz

fig2 = plt.figure('fig2')
for i in looper:
    plt.semilogy(spectrum[i].frq_domain[s_ind:f_ind],spectrum[i].avg_spect[s_ind:f_ind],label=str(X_Parameter[i]))
    plt.semilogy(spectrum[i].frq_domain[s_ind:f_ind],spectrum[i].avg_floor_repd[s_ind:f_ind])
plt.xlabel("frequency, Hz")
plt.ylabel("Field (T)")
plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.5)

names = ['DMT','HighFinesse']

fig3 = plt.figure('fig3')
for i in looper:
    plt.semilogy(spectrum[i].frq_domain,spectrum[i].avg_spect,label=names[i])
    plt.semilogy(spectrum[i].frq_domain,spectrum[i].avg_floor_repd)
plt.xlabel("frequency, Hz")
plt.ylabel("Field (T)")
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.legend()

#%%Calculate Sensitivity 
g=1/2
hbar=1.05e-34
mu=9.27e-24
redone_width = resonance.width[runs]
redone_width_err = resonance.width_err[runs]

sensitivity = np.zeros(len(looper)) 
sens_er = np.zeros(len(looper))

for i in looper:
    sensitivity[i]=(2*math.pi*redone_width[i]*hbar)/(g*mu*spectrum[i].avg_SNr)
    
    sens_er[i] = (2*math.pi*redone_width_err[i]*hbar)/(g*mu*spectrum[i].avg_SNr)
#%% Plotting
fig4= plt.figure('fig4')

#Parameter vs Width
plt.subplot(2,1,1)
plt.errorbar(X_Parameter,resonance.width[runs],yerr=resonance.width_err[runs],fmt='.b')
plt.ylim(HCA.limitise(resonance.width[runs],0.1))
plt.xlim(HCA.limitise(X_Parameter,0.1))
plt.ylabel('Width (Hz)')
plt.grid(color='k', linestyle='-', linewidth=0.5)

#Parameter vs Amplitude
plt.subplot(2,1,2)
plt.errorbar(X_Parameter,resonance.amplitude[runs],yerr=resonance.amplitude_err[runs],fmt='.b')
plt.ylim(HCA.limitise(resonance.amplitude[runs],0.1))
plt.xlim(HCA.limitise(X_Parameter,0.1))
plt.ylabel('Fit Amplitude (mV)')
plt.grid(color='k', linestyle='-', linewidth=0.5)

#Slope of resonance height/width
Fig5 = plt.figure('fig5')
plt.errorbar(X_Parameter,resonance.h_over_w[runs],yerr=resonance.h_over_w_err[runs],fmt='.b')
plt.ylim(HCA.limitise(resonance.h_over_w[runs], 0.1))
plt.xlim(HCA.limitise(X_Parameter,0.1))
plt.ylabel('Amplitude over width (mV/Hz)')
plt.xlabel(Param_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)

#Sensitivity
Fig6 = plt.figure('fig6')
plt.errorbar(X_Parameter,sensitivity,yerr=sens_er,fmt='.b')
plt.ylim(HCA.limitise_werr(sensitivity,sens_er,0.1))
plt.xlim(HCA.limitise(X_Parameter,0.1))
plt.ylabel('Sensitivity (T/rHz)')
plt.xlabel(Param_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)

#%% Extract important data
std_dev = np.zeros(len(looper))

for i in looper:
    std_dev[i] = np.std(tracking[i].cleaned_chunked_field[0])


