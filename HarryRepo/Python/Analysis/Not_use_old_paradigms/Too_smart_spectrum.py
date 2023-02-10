# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:20:41 2023

@author: vpixx
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:05:06 2023

@author: H

Looking at the DC modulation of the gradient overtop of a DC magnetic field on the gradiometer
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
# plt.style.use('classic')
import math
import Harry_analysis as HCA
from scipy.fft import fft, fftfreq

#%%
Param = [8,26,24,32,40,48,56,80] # Bz DC offset in nT
csv_sep = ';'

looper = range(len(Param))



daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230209_145756_08/freqs_000/'

track_i, track_legends = HCA.DAQ_tracking_read(daq, csv_sep)

track = HCA.DAQ_Tracking_PURE(track_i,track_legends)


#%%
t_steps = np.array([0,1,2,3,4,5,6,7])
interval = np.array([0.250,0.8])

t_ROI = np.zeros((len(interval),len(t_steps)))

ind = np.zeros((len(interval),len(t_steps)))

for i in range(len(interval)):
    t_ROI[i,:] = t_steps+interval[i]
    for j in range(len(t_steps)):
        ind[i,j] = (np.abs(track.chunked_time[0,:] - t_ROI[i,j])).argmin()
       
sind = ind[0,:]
find = ind[1,:]

T = track.chunked_time[0,1]-track.chunked_time[0,0]

solo_freq = track.chunked_data[:,int(sind[0]):int(find[0])]
flat_freq = solo_freq.flatten()

def absolutify(N,T,data_freq):
    
    xf = fftfreq(N, T)[:N//2]
   
    yf_i = fft(data_freq*0.071488e-9)
    
    yf = (2/N)*abs(np.abs(yf_i[0:N//2]))

    return xf,yf

xf_8,yf_8 = absolutify(flat_freq.shape[0], T, flat_freq)
    
plt.figure()
plt.semilogy(xf_8,yf_8)
plt.xlim(-10,100)






#%%Resonances

sfreq = 800 # Frequency of interest

#Resonance Data
fig1 = plt.figure('fig1')

Folderpath_m = daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230209_145756_08/freq_res_000/'
res_m, res_legends_m = HCA.Res_read(Folderpath_m, csv_sep)

resonance_m = HCA.Resonance(res_m, res_legends_m, sfreq)

resonance_m.plot_with_fit()









    