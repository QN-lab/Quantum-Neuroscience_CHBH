# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:22:41 2023

@author: hxc214
"""

# import obs
from Proc import obs
import matplotlib.pyplot as plt

import numpy as np
import regex as re
import pandas as pd
import os
import math
# import Harry_analysis as HCA
from scipy.fft import fft, fftfreq

# plt.rcParams['text.usetex'] = True
# plt.style.use('classic')

Gain = 14.2
T1=0
T2=3

# base_directory = 'Z:\\jenseno-opm\\Data\\2023_08_24_bench\\'
base_directory = 'Z:\\Data\\2023_08_24_bench\\'

print('Loading.....')
Data_g_on = obs.Joined(base_directory,'grad_long_run_DC_on_000', Gain, T1, T2) #2 or no 2???'
Data_g_off = obs.Joined(base_directory,'grad_long_run_DC_off_000', Gain, T1, T2)
Data_m_on = obs.Joined(base_directory,'mag_long_run_DC_on_000', Gain, T1, T2)
Data_m_off = obs.Joined(base_directory,'mag_long_run_DC_off_000', Gain, T1, T2)

#%%

avg_gon = np.mean(Data_g_on.PiD.yf_full[0,20:480])
avg_goff = np.mean(Data_g_off.PiD.yf_full[0,20:480])

avg_mon = np.mean(Data_m_on.PiD.yf_full[0,20:480])
avg_moff = np.mean(Data_m_off.PiD.yf_full[0,20:480])
# #Not right beacaus of the overlap between runs.
# fig,ax = plt.subplots(2,1,constrained_layout=True)

# fig.suptitle('Comparing Noise contribution of signal generator',fontsize=15)

# ax[0].plot(Data_g_on.PiD.xf_full[:1001],Data_g_on.PiD.yf_full[0,:1001],label='G DC on')
# ax[0].plot(Data_g_off.PiD.xf_full[:1001],Data_g_off.PiD.yf_full[0,:1001],label='G DC discon')
# ax[0].axhline(y=avg_gon,c='b',label=str(round(avg_gon/1e-15,0))+'fT/rHz')
# ax[0].axhline(y=avg_goff,c='g',label=str(round(avg_goff/1e-15,0))+'fT/rHz')
# ax[0].legend()
# ax[0].set_xlim((0,100))
# ax[0].set_yscale('log')
# # ax[0].set_xscale('log')
# ax[0].set_ylabel('Sensitivity (T/rHz)')
# ax[0].set_ylim((10**-15,10**-11))

# ax[1].plot(Data_m_on.PiD.xf_full[:1001],Data_m_on.PiD.yf_full[0,:1001],label='M DC on')
# ax[1].plot(Data_m_off.PiD.xf_full[:1001],Data_m_off.PiD.yf_full[0,:1001],label='M DC discon')
# ax[1].axhline(y=avg_mon,c='b',label=str(round(avg_mon/1e-15,0))+'fT/rHz')
# ax[1].axhline(y=avg_moff,c='g',label=str(round(avg_moff/1e-15,0))+'fT/rHz')
# ax[1].legend()
# ax[1].set_xlim((0,100))
# ax[1].set_yscale('log')
# # ax[1].set_xscale('log')
# ax[1].set_ylabel('Amplitude (T)')
# ax[1].set_xlabel('Frequency (Hz)')
# ax[1].set_ylim((10**-15,10**-11))

# #10s spectrum:
# ns = Data_g_off.PiD.yf_full[0,:1001]
# # xx = Data_g_off.PiD.xf_full[0,:1001]
# # Getting Sensitivity
# # inx = [200,400,600,1001]    #indeces to average over to get sensitivity
# inx = [800,900,901,950]

# to_avg = np.concatenate((ns[inx[0]:inx[1]],ns[inx[2]:inx[3]]))

# sensitivity = np.mean(to_avg)
# sens_ft = sensitivity/1e-15

# fig2,ax2 = plt.subplots()

# ax2.plot(Data_g_off.PiD.xf_full[:1001],Data_g_off.PiD.yf_full[0,:1001])
# ax2.set_xlim((0,100))
# ax2.grid(True)
# fig2.set_figwidth(12)
# ax2.set_yscale('log')
# # ax[0].set_xscale('log')
# ax2.set_ylabel('Noise Level ($T \: Hz^{-1/2}$)')
# ax2.set_ylim((10**-15,10**-11))
# ax2.axhline(y=sensitivity,c='g',label = 'Sensitivity= ' + str(round(sens_ft)) + '$fT \: Hz^{-1/2}$' )
# ax2.legend()
# ax2.set_xlabel('Frequency($Hz$)')
# # ax2.set_title('10s fft of Gradiometer with no applied field')


# #3s spectrum (changed to T1+3 in obs)

#%% 

run = 10
ns = Data_g_off.PiD.yf_chunked_a[run,:]
#Getting Sensitivity
# inx= [81,180,221,401]    #20-45,55-100 indeces to average over to get sensitivity
inx = [4,138,168,295] #80-100

to_avg = np.concatenate((ns[inx[0]:inx[1]],ns[inx[2]:inx[3]]))

sensitivity = np.mean(to_avg)
sens_ft = sensitivity/1e-15

fig2,ax2 = plt.subplots()

ax2.plot(Data_g_off.PiD.xf_a,Data_g_off.PiD.yf_chunked_a[run,:])
fig2.set_figwidth(12)
ax2.set_xlim((0,100))
ax2.set_yscale('log')
ax2.grid(True)
# ax[0].set_xscale('log')
ax2.set_ylabel('Noise Level ($T \: Hz^{-1/2}$)')
ax2.set_ylim((10**-15,10**-11))
ax2.axhline(y=sensitivity,c='g',label = 'Sensitivity= ' + str(round(sens_ft)) + '$fT \: Hz^{-1/2}$' )
ax2.legend()
ax2.set_xlabel('Frequency($Hz$)')
# ax2.set_title('1s fft of Gradiometer with no applied field')



# #%%



# fig3,ax3 = plt.subplots()
# ax3.set_yscale('log')
# ax3.plot(Data_g_off.PiD.xf_a[:301],Data_g_off.PiD.yf_avg_a[:301])


#%% Average across trials

# 3 SECOND SHOTS

# ns = Data_g_off.PiD.yf_chunked_a[:30,:]

# testx = Data_g_off.PiD.xf_a
# testy = np.mean(ns,axis=0)

# plt.figure()
# plt.plot(testx,testy)
# plt.xscale('log')
# plt.yscale('log')

# plt.figure()
# plt.plot(testx[7:294],testy[7:294]) #Averaged area
# plt.yscale('log')
# plt.ylim(1e-14,1e-10)
# plt.plot(testx[:7],testy[:7],'r') #non-considered area
# plt.plot(testx[294:308],testy[294:308],'r') #^
# plt.plot(testx[145:156],testy[145:156],'r') #^

# inx1 = [7,145]
# inx2 = [156,294]

# set1 = np.mean(ns[:,inx1[0]:inx1[1]],axis=1)
# av1 = np.mean(set1)/1e-15
# std1 = np.std(set1)/1e-15

# set2 = np.mean(ns[:,inx2[0]:inx2[1]],axis=1)
# av2 = np.mean(set2)/1e-15
# std2 = np.std(set2)/1e-15

#%% ONE SECOND SHOTS

ns = Data_g_off.PiD.yf_chunked_a[:30,:]

testx = Data_g_off.PiD.xf_a
testy = np.mean(ns,axis=0)

plt.figure()
plt.plot(testx,testy)
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(testx[2:98],testy[2:98]) #Averaged area
plt.yscale('log')
plt.ylim(1e-14,1e-10)
plt.plot(testx[:2],testy[:2],'r') #non-considered area
plt.plot(testx[48:52],testy[48:52],'r') #^
plt.plot(testx[98:102],testy[98:102],'r') #^
plt.axhline(y=51e-15,c='g')

inx1 = [2,48]
inx2 = [52,98]

set1 = np.mean(ns[:,inx1[0]:inx1[1]],axis=1)
av1 = np.mean(set1)/1e-15
std1 = np.std(set1)/1e-15

set2 = np.mean(ns[:,inx2[0]:inx2[1]],axis=1)
av2 = np.mean(set2)/1e-15
std2 = np.std(set2)/1e-15







