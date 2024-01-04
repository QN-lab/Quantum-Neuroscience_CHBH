# -*- coding: utf-8 -*-
"""

@author: H

"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import math
import Harry_analysis as HCA

#%% preamble & User inputs
csv_sep = ';' #separator for saved CSV
sfreq = 1500 #frequency at which to start the fitting

#%% Data Read-In

# #Resonance Data
Folderpath_r = 'C:\\Users\\vpixx\\Documents\\Zurich Instruments\\LabOne\\WebServer\\session_20230510_113045_01\\increasing_bias_3_000\\' #Resonance Data
res, res_legends = HCA.Res_read(Folderpath_r, csv_sep)

#Create resonance object
resonance = HCA.Resonance(res, res_legends, sfreq)

#Access plotting function from resonance object to fit and plot etc. 
fig1 = plt.figure('fig1')
resonance.plot_with_fit()

plt.figure()
plt.errorbar(resonance.run_names,resonance.width,yerr=resonance.width_err,fmt='.b')
plt.xlim(0,1)
plt.xlabel('Applied voltage to bias (V)')
plt.ylabel('Width (Hz)')
plt.grid()

plt.figure()
plt.errorbar(resonance.run_names,resonance.amplitude,yerr=resonance.amplitude_err,fmt = '.b')
plt.xlim(0,1)
plt.xlabel('Applied voltage to bias (V)')
plt.ylabel('Amplitude(Hz)')
plt.grid()

plt.figure()
plt.errorbar(resonance.run_names,resonance.h_over_w,yerr=resonance.h_over_w_err,fmt = '.b')
plt.xlim(0,1)
plt.xlabel('Applied voltage to bias (V)')
plt.ylabel('Slope of amplitude over width (mV/Hz)')
plt.grid()


plt.figure()
plt.errorbar(resonance.run_names,resonance.central_f,yerr=resonance.central_f_err,fmt = '.b')
plt.xlim(0,1)
plt.xlabel('Applied voltage to bias (V)')
plt.ylabel('Slope of amplitude over width (mV/Hz)')
plt.grid()

