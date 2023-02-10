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
X_Parameter = [8,16,24,32,40,48,56,80]
Param_name = 'Frequency'

#%% Data Read-In

#Resonance Data
res_loc_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230210_094924_10/grad_res_freq_1_000/' #Resonance Data
res_loc_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230210_094924_10/mag_res_freq_1_000/' #Resonance Data

res_m, res_legends_m = HCA.Res_read(res_loc_m, csv_sep)
res_g, res_legends_g = HCA.Res_read(res_loc_g, csv_sep)

resonance_m = HCA.Resonance(res_m, res_legends_m,sfreq)
resonance_g = HCA.Resonance(res_g, res_legends_g,sfreq)

fig1 = plt.figure('fig1')
resonance_m.plot_with_fit()
resonance_g.plot_with_fit()

#DAQ Data
daq_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230210_094924_10/mag_freqs_000/'
daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230210_094924_10/grad_freqs_000/'

spect_m, spect_legends_m = HCA.DAQ_read_xiy(daq_m,csv_sep)
spect_g, spect_legends_g = HCA.DAQ_read_xiy(daq_g,csv_sep)

spectrum_m = HCA.DAQ_Spectrum_none(spect_m,spect_legends_m)
spectrum_g = HCA.DAQ_Spectrum_none(spect_g,spect_legends_g)

#%%plot ROI on spectrum
for i in spectrum_m.patch:
    plt.figure()
    plt.semilogy(spectrum_m.frq_domain,spectrum_m.chunked_data[i])
    plt.semilogy(spectrum_g.frq_domain,spectrum_g.chunked_data[i])
    plt.semilogy(spectrum_m.frq_domain,spectrum_m.floor_repd[i])
    plt.semilogy(spectrum_m.frq_domain,spectrum_m.floor_repd[i])
plt.xlabel("frequency, Hz")
plt.ylabel("quadrature, mV")
plt.grid(color='k', linestyle='-', linewidth=0.5)

#%%Calculate Sensitivity 
g=1/2
hbar=1.05e-34
mu=9.27e-24
sensitivity_m=(2*math.pi*resonance_m.width*hbar)/(g*mu*spectrum_m.SNr)
sens_er_m = (2*math.pi*resonance_m.width_err*hbar)/(g*mu*spectrum_m.SNr) #assumes no error on SNr, probably need to account for this


sensitivity_g=(2*math.pi*resonance_g.width*hbar)/(g*mu*spectrum_g.SNr)
sens_er_g = (2*math.pi*resonance_g.width_err*hbar)/(g*mu*spectrum_g.SNr) #assumes no error on SNr, probably need to account for this

#%% Plotting

#Sensitivity
plt.figure()
plt.plot([1,2,3,4,5,6,7,8],sensitivity_m/1e-15,'o',label = 'mag')
plt.plot([1,2,3,4,5,6,7,8],sensitivity_g/1e-15,'o',label = 'grad')
plt.ylabel('Sensitivity (fT/rHz)')
plt.xlabel('run number')
plt.legend()
plt.xlim(-1,9)
plt.grid(color='k', linestyle='-', linewidth=0.2)


plt.figure()
plt.plot([1,2,3,4,5,6,7,8],sensitivity_g/4e-15,'go',label = 'grad')
plt.ylabel('Sensitivity (fT/cm*rHz)')
plt.xlabel('run number')
plt.legend()
plt.xlim(-1,9)
plt.grid(color='k', linestyle='-', linewidth=0.2)

#Averages

sens_m = np.mean(sensitivity_m/1e-15)
sens_g = np.mean(sensitivity_g/1e-15)

sens_g_bl = np.mean(sensitivity_g/4e-15)



