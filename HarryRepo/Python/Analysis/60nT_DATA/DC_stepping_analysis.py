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
Param = [0,0.2,0.4,0.5,0.6,0.8,1] # Bz DC offset in nT, 10 removed cause it was too much
csv_sep = ';'

looper = range(len(Param))

track = list([])

daq_m = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230213_142803_01/mag_C1_DC_steps_2_000/'
daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230213_142803_01/grad_DC_steps_2_000/'

track_i_m, track_legends_m = HCA.DAQ_read_shift(daq_m, csv_sep)
track_i_g, track_legends_g = HCA.DAQ_read_shift(daq_g, csv_sep)

track_m = HCA.DAQ_Tracking_PURE(track_i_m,track_legends_m)
track_g = HCA.DAQ_Tracking_PURE(track_i_g,track_legends_g)

for j in range(10):
    plt.figure()
    plt.plot(track_m.chunked_time[j],track_m.chunked_data[j])
    plt.ticklabel_format(useOffset=False)
    # plt.ylim(1656.5,1659)
    plt.xlabel('Time (s)')
    plt.ylabel('Measured magnetic field in Modulation freq shift (Hz)')
    plt.title('Magnetometer DC test, Run No.' + str(j+1))
    # plt.legend(title='Z DC offset (nT)', fontsize=10)

for j in range(10):
    plt.figure()
    plt.plot(track_g.chunked_time[j],(track_g.chunked_data[j])/4)
    plt.ticklabel_format(useOffset=False)
    # plt.ylim(1656.5,1659)
    plt.xlabel('Time (s)')
    plt.ylabel('Measured gradient in Modulation freq shift (Hz)')
    plt.title('Magnetometer DC test, Run No.' + str(j+1))
    # plt.legend(title='Z DC offset (nT)', fontsize=10)

#%%
t_steps = np.array([0,1,2,3,4,5,6])
interval = np.array([0.250,0.8])

t_ROI = np.zeros((len(interval),len(t_steps)))
ind = np.zeros((len(interval),len(t_steps)))

for i in range(len(interval)):
    t_ROI[i,:] = t_steps+interval[i]
    for j in range(len(t_steps)):
        ind[i,j] = (np.abs(track_m.chunked_time[0,:] - t_ROI[i,j])).argmin() # chunked time should be the same for all runs
   
sind = ind[0,:]
find = ind[1,:]


def pull_mean_vals(sind,find,looper,Param,chunked_data):
    
    mean_vals = np.zeros((10,len(Param)))
    for i in range(10):
        for j in range(len(Param)):
            mean_vals[i,j] = np.mean(chunked_data[i,int(sind[j]):int(find[j])])
        
    return mean_vals
        

vals_m = pull_mean_vals(sind,find,looper,Param,track_m.chunked_data)
vals_g = pull_mean_vals(sind,find,looper,Param,track_g.chunked_data)
        
        
plt.figure()
for i in range(10):
    plt.plot(Param,vals_m[i,:],'-o', label = str(i+1))
plt.xlabel('Applied HG Field (nT)')
plt.ylabel('Measured field in Modulation freq shift (Hz)')
plt.ticklabel_format(useOffset=False)
plt.legend(title='Run no.',fontsize=10,loc=2)
plt.title('Magnetometer DC test')
plt.grid()

plt.figure()
for i in range(10):
    plt.plot(Param,vals_g[i,:],'-o', label = str(i+1))
plt.xlabel('Applied HG Field (nT)')
plt.ylabel('Measured gradient in Modulation freq shift (Hz)')
plt.ticklabel_format(useOffset=False)
plt.legend(title='Run no.',fontsize=10,loc=2)
plt.title('Gradiometer DC test')
plt.grid()


#Attenuation (Average within each field)

atten = np.zeros(len(looper))
for i in looper:
    atten[i] = np.abs(np.mean(vals_m[:,i]))/(np.abs(np.mean(vals_g[:,i])))
    
plt.figure()
plt.plot(Param,atten,'o')
plt.xlabel('Applied HG Field (nT)')
plt.ylabel('Attenuation (arb)')
plt.ticklabel_format(useOffset=False)
# plt.legend(title='Run no.',fontsize=10,loc=2)
plt.title('Attenuation of gradiometer against Cell 1')
# plt.ylim(0,50)
plt.xlim(-0.2,1.2)
plt.grid()

#avg avross runs
vals_m_avg = np.mean(vals_m,axis=0)
vals_g_avg = np.mean(vals_g,axis=0)


coefs = np.polyfit(Param,vals_m[0,:],1)




#%%Resonances

sfreq = 800 # Frequency of interest

#Resonance Data

daq_m1 = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230213_142803_01/mag_C1_DC_steps_res_2_000/'
res_m1, res_legends_m1 = HCA.Res_read(daq_m1, csv_sep)

resonance_m1 = HCA.Resonance(res_m1, res_legends_m1, sfreq)

daq_m2 = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230213_142803_01/mag_C2_DC_steps_res_2_000/'
res_m2, res_legends_m2 = HCA.Res_read(daq_m2, csv_sep)

resonance_m2 = HCA.Resonance(res_m2, res_legends_m2, sfreq)
plt.figure('fig1')


fig1 = plt.figure('fig1')

resonance_m1.plot_with_fit()
resonance_m2.plot_with_fit()






    