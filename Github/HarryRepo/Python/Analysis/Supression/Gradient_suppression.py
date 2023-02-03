# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:50:48 2023

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import math
import Harry_analysis as HCA

#%% preamble & User inputs
csv_sep = ';' #separator for saved CSV
sfreq = 800 #frequency at which to start the fitting
X_Parameter = [0,0.5,1,2,5,10,15]
Param_name = 'Added noise (nT)'

#%% Data Read-In

#Resonance Data
Folderpath_r = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230125_145005_01/grad_res_1_000/' #Resonance Data
res, res_legends = HCA.Res_read(Folderpath_r, csv_sep)

#Create resonance object
resonance1 = HCA.Resonance(res, res_legends, sfreq)

#Access plotting function from resonance object to fit and plot etc. 
fig1 = plt.figure('fig1')
resonance1.plot_with_fit()

#Resonance Data
Folderpath_r = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230125_145005_01/grad_res_1_001/' #Resonance Data
res, res_legends = HCA.Res_read(Folderpath_r, csv_sep)

#Create resonance object
resonance2 = HCA.Resonance(res, res_legends, sfreq)

#Access plotting function from resonance object to fit and plot etc. 
fig1 = plt.figure('fig1')
resonance2.plot_with_fit()


#DAQ Data
trigger1 = list([])
tracking1 = list([])
spectrum1 = list([])

trigger2 = list([])
tracking2 = list([])
spectrum2 = list([])

looper = range(len(X_Parameter))
#Create lists of objects, each corresponding to every DAQ run
for i in looper:
    daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230125_145005_01/{}_nT_noise_000/'.format(X_Parameter[i])
    
    trig, trig_legends = HCA.DAQ_trigger_read(daq,csv_sep)
    track, track_legends = HCA.DAQ_tracking_read(daq,csv_sep)
    spect, spect_legends = HCA.DAQ_spect_read(daq,csv_sep)
    
    #Lists of objects
    trigger1.append(HCA.DAQ_Trigger(trig,trig_legends))
    tracking1.append(HCA.DAQ_Tracking(track,track_legends,trigger1[i]))
    spectrum1.append(HCA.DAQ_Spectrum(spect,spect_legends,trigger1[i]))
    
    daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230125_145005_01/{}_nT_noise_001/'.format(X_Parameter[i])
    
    trig, trig_legends = HCA.DAQ_trigger_read(daq,csv_sep)
    track, track_legends = HCA.DAQ_tracking_read(daq,csv_sep)
    spect, spect_legends = HCA.DAQ_spect_read(daq,csv_sep)
    
    #Lists of objects
    trigger2.append(HCA.DAQ_Trigger(trig,trig_legends))
    tracking2.append(HCA.DAQ_Tracking(track,track_legends,trigger2[i]))
    spectrum2.append(HCA.DAQ_Spectrum(spect,spect_legends,trigger2[i]))

    
    
fig200 = plt.figure('fig200')

i = 1  #500pT noise

plt.semilogy(spectrum1[1].frq_domain, spectrum1[1].single_spect, label='gradiometer')
plt.semilogy(spectrum2[1].frq_domain, spectrum2[1].single_spect, label='magnetometer')

plt.semilogy(spectrum1[1].frq_domain,spectrum1[1].single_floor_repd,'k-')
plt.semilogy(spectrum2[1].frq_domain,spectrum2[1].single_floor_repd,'k-')
plt.legend()
plt.title('mag vs grad at 0.5nT white noise')
plt.xlabel("frequency, Hz")
plt.ylabel("Power spectral density (V/rHz)")
plt.grid(color='k', linestyle='-', linewidth=0.5)
    
qq = spectrum1[1].single_SNr/spectrum2[1].single_SNr
    

#%%Calculate Sensitivity 
g=1/2
hbar=1.05e-34
mu=9.27e-24

sensitivity = np.zeros(len(looper)) 
sens_er = np.zeros(len(looper))

for i in looper:
    sensitivity[i]=(2*math.pi*resonance.width*hbar)/(g*mu*spectrum[i].single_SNr) #4 is the 4cm baseline
    sens_er[i] = (2*math.pi*resonance.width*hbar)/(g*mu*spectrum[i].single_SNr)
    
#Sensitivity
Fig300 = plt.figure('fig300')
plt.plot(X_Parameter,sensitivity/1e-15,'k*')
plt.xlim(-1,16)
plt.ylabel('Sensitivity (fT/rHz)')
plt.xlabel(Param_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)


