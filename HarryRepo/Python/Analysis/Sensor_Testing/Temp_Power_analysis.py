# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:57:22 2023

@author: H

Writing the code setup for comparing parameters
"""

#TO Do
#Error on both parameters

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
import math
import Harry_analysis as HCA

#%% preamble & User inputs
csv_sep = ';' #separator for saved CSV
sfreq = 1400 #frequency domain value at which to start the fitting
Temp_Param = [30,32,34,36,38,40,42]
Temp_name = 'Temperature (C)'
T_err = -0.4

Power_Param = [80,100,120,140,160,180,200]
P_err = 3
Power_name = 'Power (uW)'

#%% Data Read-In

looper_T = range(len(Temp_Param))
looper_P = range(len(Power_Param))

resonance = list([])
#Resonance Data
for i in looper_T:
    
    Folderpath_r = 'Z:/jenseno-opm/Data/2023_01_20/{}_res_000/'.format(Temp_Param[i]) #Resonance Data
    res, res_legends = HCA.Res_read(Folderpath_r, csv_sep)
    
    resonance.append(HCA.Resonance(res, res_legends, sfreq))

#DAQ Data
trigger = list([])
tracking = list([])
spectrum = list([])

trigger_temp = list([])
tracking_temp = list([])
spectrum_temp = list([])

#Create lists of objects, each corresponding to every DAQ run
for i in looper_T:
    for j in looper_P:
        daq = 'Z:/jenseno-opm/Data/2023_01_20/{}_{}_daq_000/'.format(Temp_Param[i],Power_Param[j])
        
        trig, trig_legends = HCA.DAQ_trigger_read(daq,csv_sep)
        track, track_legends = HCA.DAQ_tracking_read(daq,csv_sep)
        spect, spect_legends = HCA.DAQ_spect_read(daq,csv_sep)
        
        #Lists of objects
        trigger_temp.append(HCA.DAQ_Trigger(trig,trig_legends))
        tracking_temp.append(HCA.DAQ_Tracking(track,track_legends,trigger_temp[j]))
        spectrum_temp.append(HCA.DAQ_Spectrum(spect,spect_legends,trigger_temp[j]))
        
    #List of lists
    trigger.append(trigger_temp)
    tracking.append(tracking_temp)
    spectrum.append(spectrum_temp)
    trigger_temp = list([])
    tracking_temp = list([])
    spectrum_temp = list([])

# #plotting resonances
fig1= plt.figure('fig1')

for i in looper_T: 
    fig= plt.figure('fig' + str(i+1))
    resonance[i].plot_with_fit()
    plt.title('Temperature: {}C'.format(Temp_Param[i]))
    plt.legend(title=Power_name)
    plt.ylim(-1,5)
    
#%% Spectra (Tempwise)

# for j in looper_T:
#     fig= plt.figure('fig' + str(j+1))
#     plt.title('Temperature: {}C'.format(Temp_Param[j]))
#     for i in looper_P: 
#         plt.semilogy(spectrum[j][i].frq_domain,spectrum[j][i].avg_spect)
#         plt.semilogy(spectrum[j][i].frq_domain,spectrum[j][i].avg_floor_repd)
#         plt.xlabel("frequency, Hz")
#         plt.ylabel("Power spectral density (V/rHz)")
#         plt.xlim(HCA.limitise(spectrum[j][i].frq_domain, 0.2))
#         plt.grid(color='k', linestyle='-', linewidth=0.5)

#%% Resonance parameter analysis

fig300= plt.figure('fig300')
Temp = np.zeros(len(looper_T))
Temp_err =np.zeros(len(looper_T))
#Parameter vs Width
for i in looper_P:
    for j in looper_T:
        Temp[j] = resonance[j].width[i]
        Temp_err[j] = resonance[j].width_err[i]
        
    plt.errorbar(Temp_Param,Temp,yerr=Temp_err,xerr=T_err, 
                 fmt = '.',xlolims=True, label = str(Power_Param[i]))
    Temp = np.zeros(len(looper_T))
    
plt.legend(title=Power_name)
plt.ylim(30,90)
plt.xlim(HCA.limitise(Temp_Param,0.1))
plt.ylabel('Width (Hz)')
plt.xlabel(Temp_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)


fig400 = plt.figure('fig400')
#Parameter vs Amplitude
for i in looper_P:
    for j in looper_T:
        Temp[j] = resonance[j].amplitude[i]
        Temp_err[j] = resonance[j].amplitude_err[i]
        
    plt.errorbar(Temp_Param,Temp,yerr=Temp_err,xerr=T_err, 
                 fmt = '.',xlolims=True, label = str(Power_Param[i]))
    Temp = np.zeros(len(looper_T))
    
plt.legend(title=Power_name)
plt.ylim(0,5)
plt.xlim(HCA.limitise(Temp_Param,0.1))
plt.ylabel('Amplitude (mV))')
plt.xlabel(Temp_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)

fig500 = plt.figure('fig500')
#Slope of resonance height/width
for i in looper_P:
    for j in looper_T:
        Temp[j] = resonance[j].h_over_w[i]
        Temp_err[j] = resonance[j].h_over_w_err[i]
        
    plt.errorbar(Temp_Param,Temp,yerr=Temp_err,xerr=T_err, 
                 fmt = '.',xlolims=True, label = str(Power_Param[i]))
    Temp = np.zeros(len(looper_T))
    
plt.legend(title=Power_name)
# plt.ylim(30,90)
plt.xlim(HCA.limitise(Temp_Param,0.1))
plt.ylabel('Amplitude/Width (mV/Hz))')
plt.xlabel(Temp_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)


#%% Sensitivity analysis

g=1/2
hbar=1.05e-34
mu=9.27e-24

sensitivity = np.zeros((len(looper_T),len(looper_P))) 
sens_er = np.zeros((len(looper_T),len(looper_P))) 

for j in looper_T:
    for i in looper_P:
        sensitivity[j,i]=(2*math.pi*resonance[j].width[i]*hbar)/(g*mu*spectrum[j][i].avg_SNr)
        sens_er[j,i] = (2*math.pi*resonance[j].width_err[i]*hbar)/(g*mu*spectrum[j][i].avg_SNr)
        
fig600 = plt.figure('fig600')

for i in looper_P: 
        plt.errorbar(Temp_Param,sensitivity[:,i]/1e-15,yerr=sens_er[:,i],xerr=T_err,
                     fmt = '-',xlolims=True,label = str(Power_Param[i]))
        plt.legend(title=Power_name)
plt.xlim(HCA.limitise(Temp_Param,0.1))
plt.ylabel('Sensitivity (fT/rHz)')
plt.xlabel(Temp_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)


#%% Investigation

#looking for an average trend in sensitivity vs temp

tempwise_sens = np.mean(sensitivity,axis=1) #average over axis 1: power

fig700 = plt.figure('fig700')

plt.errorbar(Temp_Param,tempwise_sens/1e-15,xerr = T_err, xlolims=True, fmt='b.')

plt.xlim(HCA.limitise(Temp_Param,0.1))
plt.ylabel('Power-Averaged Sensitivity (fT/rHz)')
plt.ylim(HCA.limitise(tempwise_sens/1e-15,0.2))
plt.xlabel(Temp_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)

#Power vs sensitivity

powerwise_sens = np.mean(sensitivity,axis=0) #average over axis 0: temperature

fig800 = plt.figure('fig800')

plt.errorbar(Power_Param,powerwise_sens/1e-15,xerr = P_err, fmt='b.')

plt.xlim(HCA.limitise(Power_Param,0.1))
plt.ylabel('Temperature-Averaged Sensitivity (fT/rHz)')
plt.ylim(HCA.limitise(powerwise_sens/1e-15,0.2))
plt.xlabel(Power_name)
plt.grid(color='k', linestyle='-', linewidth=0.5)









