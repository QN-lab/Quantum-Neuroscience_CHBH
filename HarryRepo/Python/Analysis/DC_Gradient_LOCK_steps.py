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

#%%
Param = [0,1,5,10,30,60] # Bz DC offset in nT
ref_grad = [0,0.5,1,1.5,2,5,10,20] #pT/cm
csv_sep = ';'

looper = range(len(Param))

track = list([])

for i in looper:
    daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230208_233828_07/{}_DC_000/'.format(Param[i])
    
    track_i, track_legends = HCA.DAQ_read_shift(daq, csv_sep)
    
    track.append(HCA.DAQ_Tracking_PURE(track_i,track_legends))
    
#Raw signal (run 1)

for j in range(5):
    plt.figure()
    for i in looper:
        plt.plot(track[i].chunked_time[j],track[i].chunked_data[j],label = str(Param[i]))
    plt.ticklabel_format(useOffset=False)
    # plt.ylim(1656.5,1659)
    plt.xlabel('Time (s)')
    plt.ylabel('Measured gradient in Modulation freq shift (Hz)')
    plt.title('Gradiometer DC test, Run No.' + str(j+1))
    plt.legend(title='Z DC offset (nT)', fontsize=10)
    
# plt.figure()
# for j in range(10):
#     for i in looper:
#         plt.plot(aux[2].chunked_data[j])  #track[2].chunked_time[j],
#     plt.ticklabel_format(useOffset=False)
#     # plt.ylim(1656.5,1659)
#     plt.xlabel('Time Samples') #'Time (s)'
#     plt.ylabel('Modulation Frequency of Gradiometer (Hz)')
#     # plt.title('Gradiometer DC test, Run No.' + str(j+1))
    
    
#%%
t_steps = np.array([0,1,2,3,4,5,6,7])
interval = np.array([0.250,0.8])

t_ROI = np.zeros((len(interval),len(t_steps)))

ind = np.zeros((len(interval),len(t_steps)))

for i in range(len(interval)):
    t_ROI[i,:] = t_steps+interval[i]
    for j in range(len(t_steps)):
        ind[i,j] = (np.abs(track[0].chunked_time[2,:] - t_ROI[i,j])).argmin()
       
sind = ind[0,:]
find = ind[1,:]

# for j in range(len(ref_grad)):
#     plt.axhline(mean_vals[j])

mean_vals = np.zeros((len(Param),len(ref_grad)))


for i in looper:
    for j in range(len(ref_grad)):
        mean_vals[i,j] = np.mean(track[i].chunked_data[2,int(sind[j]):int(find[j])])
        
plt.figure()
for i in looper:
    plt.plot(ref_grad,mean_vals[i,:],'-o',label = str(Param[i]))
plt.xlabel('Applied Gradient(pT/cm)')
plt.ylabel('Measured gradient in Modulation freq shift (Hz)')
plt.ticklabel_format(useOffset=False)
plt.legend(title='DC Mag Field (nT)',fontsize=10)
plt.title('Gradiometer DC test, Run No. 3')
plt.grid()

#Average over trials

mean_vals_mat = np.zeros((5,len(Param),len(ref_grad)))
for h in range(5):
    for i in looper:
        for j in range(len(ref_grad)):
            mean_vals_mat[h,i,j] = np.mean(track[i].chunked_data[h,int(sind[j]):int(find[j])])

mean_vals_tot = np.mean(mean_vals_mat,axis=0)

coefs = np.zeros((2,len(looper)))

plt.figure()
for i in looper:
    
    coefs[:,i] = np.polyfit(ref_grad, mean_vals_tot[i,:], 1)
    poly1d_fn = np.poly1d(coefs[:,i]) 
    plt.plot(ref_grad,mean_vals_tot[i,:],'ko')
    plt.plot(ref_grad,poly1d_fn(ref_grad),'--', label = str(Param[i]))
    
    
plt.xlabel('Applied Gradient(pT/cm)')
plt.ylabel('Measured gradient in Modulation freq shift (Hz)')
plt.ticklabel_format(useOffset=False)
plt.legend(title='DC Mag Field (nT)',fontsize=10)
plt.title('Gradiometer DC test, All runs, averaged')
plt.grid()

avg_slope = np.mean(coefs[0,:])

#%%Resonances

sfreq = 800 # Frequency of interest

#Resonance Data
fig1 = plt.figure('fig1')

Folderpath_m = daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230208_233828_07/grad_res_000/'
res_m, res_legends_m = HCA.Res_read(Folderpath_m, csv_sep)

resonance_m = HCA.Resonance(res_m, res_legends_m, sfreq)

resonance_m.plot_with_fit()









    