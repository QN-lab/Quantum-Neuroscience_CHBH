# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:22:41 2023

@author: hxc214
"""
# import obs
from Proc import obs
# import obs
import matplotlib.pyplot as plt
import numpy as np
import regex as re
import pandas as pd
import os
import math
# import Harry_analysis as HCA
from scipy.fft import fft, fftfreq

plt.rcParams['text.usetex'] = True
plt.style.use('classic')

Gain = 14.2
T1=0
T2=3
# base_directory_g = 'Z:\\jenseno-opm\\Data\\2023_08_18_bench\\grad_noise_high\\'
base_directory_g = 'Z:\\Data\\2023_08_17_bench\\grad_noise\\'
# base_directory_g = 'Z:\\Data\\2023_08_25_bench\\grad_100nT_bb_noise\\'
subfolder_list_g = os.listdir(base_directory_g)

# base_directory_m = 'Z:\\jenseno-opm\\Data\\2023_08_18_bench\\mag_noise_high\\'
base_directory_m = 'Z:\\Data\\2023_08_17_bench\\mag_noise\\'
# base_directory_m = 'Z:\\Data\\2023_08_25_bench\\mag_100nT_bb_noise\\'

subfolder_list_m = os.listdir(base_directory_m)
 

print('Loading.....')

Data_list_g = list()
for cur_subfolder in subfolder_list_g:
    print('...')
    Data_list_g.append(obs.Joined(base_directory_g, cur_subfolder, Gain, T1, T2))
    print('Loaded ' + cur_subfolder)
    
    
Data_list_m = list()
for cur_subfolder in subfolder_list_m:
    print('...')
    Data_list_m.append(obs.Joined(base_directory_m, cur_subfolder, Gain, T1, T2))
    print('Loaded ' + cur_subfolder)
#%%
#Plotting single-shot spectra of grad/mag with BB noise on and off.

ind = 2
# mV = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65])
# mV = np.array([0,3,6,9,12,15,18,21,24,27,30,33])
mV = np.array([3,6,9,12,15,18,21,24,27,30,33])
b_field = mV/6.45

run = 0

g_off = Data_list_g[0].PiD.yf_chunked_a[run,:]
g_on = Data_list_g[ind].PiD.yf_chunked_a[run,:]

m_off = Data_list_m[0].PiD.yf_chunked_a[run,:]
m_on = Data_list_m[ind].PiD.yf_chunked_a[run,:]

xf =  Data_list_g[0].PiD.xf_a


fig,ax = plt.subplots(2,1,constrained_layout=True)

fig.suptitle('Comparing grad vs mag for first value of applied BB noise',fontsize=15)

ax[0].plot(xf[:300],g_off[:300],label='grad 0nT')
ax[0].plot(xf[:300],g_on[:300],label='grad '+str(round(b_field[ind],3))+'nT')
ax[0].set_xlim((0,100))
ax[0].set_ylim((1e-14,1e-10))
ax[0].set_ylabel('Noise Amp (T/rHz)')
ax[0].set_yscale('log')
ax[0].legend()

ax[1].plot(xf[:300],m_off[:300],label = 'mag 0nT')
ax[1].plot(xf[:300],m_on[:300],label = 'mag ' +str(round(b_field[ind],3))+'nT')
ax[1].set_xlim((0,100))
ax[1].set_ylim((1e-14,1e-10))
ax[1].set_yscale('log')
ax[1].set_ylabel('Noise Amp (T/rHz)')
ax[1].legend()
ax[1].set_xlabel('Frequency (Hz)')


#%% Average floor excl. 6Hz

goc = np.delete(g_off,18)
gnc = np.delete(g_on,18)

moc = np.delete(m_off,18)
mnc = np.delete(m_on,18)

xfc = np.delete(xf,18)

plt.figure()
plt.plot(xfc,goc)
plt.plot(xfc,gnc)
plt.plot(xfc,moc)
plt.plot(xfc,mnc)
plt.yscale('log')
plt.xlim(0,100)

av_goff = np.mean(goc[1:57])
av_gon = np.mean(gnc[1:57])
av_moff = np.mean(moc[1:57])
av_mon = np.mean(mnc[1:57])

#%%one plot
plt.figure()

plt.plot(xf[:300],g_off[:300],c='b')
plt.plot(xf[:300],g_on[:300],c='g')

plt.plot(xf[:300],m_off[:300],c='r')
plt.plot(xf[:300],m_on[:300],c='c')

plt.axhline(y=av_goff,c='b',label='grad 0nT; '+str(round(av_goff/1e-15))+'fT/rHz')
plt.axhline(y=av_gon,c='g',label='grad '+str(round(b_field[ind],3))+'nT; '+str(round(av_gon/1e-15))+'fT/rHz')
plt.axhline(y=av_moff,c='r',ls='--',label = 'mag 0nT; '+str(round(av_moff/1e-15))+'fT/rHz')
plt.axhline(y=av_mon,c='c',label = 'mag ' +str(round(b_field[ind],3))+'nT; '+str(round(av_mon/1e-15))+'fT/rHz')

plt.xlim((0,100))
plt.ylim((1e-14,1e-10))
plt.ylabel('Noise Amp (T/rHz)')
plt.yscale('log')
plt.xlabel('Frequency(Hz)')
plt.legend()
plt.gca().set_aspect(12)

#%% SNR

sig_g = np.zeros((11,10))
sig_m = np.zeros((11,10))
n_g = np.zeros((11,10))
n_m = np.zeros((11,10))

for i in range(11): # noise
    for j in range(10): #run within noise 
        sig_g[i,j] = Data_list_g[i].PiD.yf_chunked_a[j,18]
        sig_m[i,j] = Data_list_m[i].PiD.yf_chunked_a[j,18]
        n_g[i,j] = np.mean(Data_list_g[i].PiD.yf_chunked_a[j,30:150])
        n_m[i,j] = np.mean(Data_list_m[i].PiD.yf_chunked_a[j,30:150])

sig_av_g = np.mean(sig_g,axis=1)
sig_std_g = np.std(sig_g,axis=1)

sig_av_m = np.mean(sig_m,axis=1)
sig_av_m = sig_av_m[0]*np.ones(len(sig_av_m))
sig_std_m = np.std(sig_m,axis=1)
sig_std_m = sig_std_m[0]*np.ones(len(sig_av_m))

n_av_g = np.mean(n_g,axis=1)
n_std_g = np.std(n_g,axis=1)

n_av_m = np.mean(n_m,axis=1)
n_std_m = np.std(n_m,axis=1)

snr_g = sig_av_g/n_av_g
snr_m = sig_av_m/n_av_m

err_g = snr_g*np.sqrt((sig_std_g/sig_av_g)**2+(n_std_g/n_av_g)**2)**0.5
err_m = snr_m*np.sqrt((sig_std_m/sig_av_m)**2+(n_std_m/n_av_m)**2)**0.5

fig,ax = plt.subplots()

ax.errorbar(b_field,snr_g,yerr=err_g,fmt='g*',label='gradiometer')
ax.errorbar(b_field,snr_m,yerr=err_m,fmt='b*',label='magnetometer')
ax.set(title='SNR',xlim=(0.35,10.5),ylabel='SNR',xlabel='Applied field (nT)',yscale='log')
ax.grid()
ax.legend()

fig,ax = plt.subplots(3,1,constrained_layout=True)

ax[0].errorbar(b_field,snr_g,yerr=err_g,fmt='g*',label='gradiometer')
ax[0].errorbar(b_field,snr_m,yerr=err_m,fmt='b*',label='magnetometer')
ax[0].set(title='SNR',xlim=(-0.5,5.5),ylabel='SNR',xlabel='Applied field (nT)',yscale='log')
ax[0].grid()

ax[1].errorbar(b_field,sig_av_g,yerr=sig_std_g,fmt='g*',label='gradiometer')
ax[1].errorbar(b_field,sig_av_m,yerr=sig_std_m,fmt='b*',label='magnetometer')
ax[1].set(title='S',xlim=(-0.5,5.5),ylabel='Signal(T/rHz)',xlabel='Applied field (nT)')


ax[2].errorbar(b_field,n_av_g,yerr=n_std_g,fmt='g*',label='gradiometer')
ax[2].errorbar(b_field,n_av_m,yerr=n_std_m,fmt='b*',label='magnetometer')
ax[2].set(title='N',xlim=(-0.5,5.5),ylabel='Noise(T/rHz)',xlabel='Applied field (nT)')
ax[2].legend(loc='upper left')

arr = [b_field,
       sig_av_g,sig_std_g,
       sig_av_m,sig_std_m,
       n_av_g,n_std_g,
       n_av_m,n_std_m,
       snr_g,err_g,
       snr_m,err_m]

ref = ['b_field',
       'sig_av_g','sig_std_g',
       'sig_av_m','sig_std_m',
       'n_av_g','n_std_g',
       'n_av_m','n_std_m',
       'snr_g','err_g',
       'snr_m','err_m']

df = pd.DataFrame(arr,ref).transpose()

