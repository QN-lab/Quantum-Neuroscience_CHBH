# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:08:19 2023

@author: h
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import math
import Harry_analysis as HCA

csv_sep = ';' #separator for saved CSV
sfreq = 800 #frequency at which to start the fitting

#Resonance Data
Folderpath_r = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230130_101502_07/mag_res_3_000/' #Resonance Data
res, res_legends = HCA.Res_read(Folderpath_r, csv_sep)

mag_res = HCA.Resonance(res, res_legends, sfreq)

Folderpath_r = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230130_101502_07/grad_res_4_000/' #Resonance Data
res, res_legends = HCA.Res_read(Folderpath_r, csv_sep)

grad_res = HCA.Resonance(res, res_legends, sfreq)

fig1 = plt.figure('fig1')
mag_res.plot_with_fit()
grad_res.plot_with_fit()

st_g = ['0pT_0pTcm_grad_4','0pT_4pTcm_grad_4','400pT_4pTcm_grad_4']
st_m = ['0pT_0pTcm_mag_3','0pT_4pTcm_mag_3','400pT_4pTcm_mag_3']

looper = range(len(st_g))

trigger1 = list([])
tracking1 = list([])
spectrum1 = list([])

for i in looper:

        daq1 = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230130_101502_07/{}_000/'.format(st_g[i])
        
        trig1, trig_legends1 = HCA.DAQ_trigger_read(daq1,csv_sep)
        track1, track_legends1 = HCA.DAQ_tracking_read(daq1,csv_sep)
        spect1, spect_legends1 = HCA.DAQ_spect_read(daq1,csv_sep)
        
        #Lists of objects
        trigger1.append(HCA.DAQ_Trigger(trig1,trig_legends1))
        tracking1.append(HCA.DAQ_Tracking(track1,track_legends1,trigger1[i]))
        spectrum1.append(HCA.DAQ_Spectrum(spect1,spect_legends1,trigger1[i]))

fig2 = plt.figure('fig2')
for i in looper:
    plt.semilogy(spectrum1[i].frq_domain,spectrum1[i].avg_spect,label = st_g[i])
    plt.semilogy(spectrum1[i].frq_domain,spectrum1[i].avg_floor_repd)
plt.xlabel("frequency, Hz")
plt.ylabel("PSD(V/rHz)")
plt.legend()
plt.xlim(-10,440)
plt.ylim(1e-9,1e-2)
plt.grid(color='k', linestyle='-', linewidth=0.5)

trigger2 = list([])
tracking2 = list([])
spectrum2 = list([])

for i in looper:

        daq2 = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230130_101502_07/{}_000/'.format(st_m[i])
        
        trig2, trig_legends2 = HCA.DAQ_trigger_read(daq2,csv_sep)
        track2, track_legends2 = HCA.DAQ_tracking_read(daq2,csv_sep)
        spect2, spect_legends2 = HCA.DAQ_spect_read(daq2,csv_sep)
        
        #Lists of objects
        trigger2.append(HCA.DAQ_Trigger(trig2,trig_legends2))
        tracking2.append(HCA.DAQ_Tracking(track2,track_legends2,trigger2[i]))
        spectrum2.append(HCA.DAQ_Spectrum(spect2,spect_legends2,trigger2[i]))


fig3 = plt.figure('fig3')
for i in looper:
    plt.semilogy(spectrum2[i].frq_domain,spectrum2[i].avg_spect,label = st_m[i])
    plt.semilogy(spectrum2[i].frq_domain,spectrum2[i].avg_floor_repd)
plt.xlabel("frequency, Hz")
plt.ylabel("PSD(V/rHz)")
plt.legend()
plt.xlim(-10,440)
plt.ylim(1e-9,1e-2)
plt.grid(color='k', linestyle='-', linewidth=0.5)



#looking at dividing one spectrum by another










