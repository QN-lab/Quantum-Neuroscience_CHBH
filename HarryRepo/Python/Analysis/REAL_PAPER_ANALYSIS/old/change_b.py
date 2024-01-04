# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:42:06 2023

@author: H
"""
from Proc import obs
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
import regex as re
import pandas as pd
import os
import math
import Harry_analysis as HCA
from scipy.fft import fft, fftfreq

###############################################################################
#%%

Gain = 14.2
T1=0
T2=3

runtype = 1

# base_directory_g = 'Z:\\Data\\2023_08_11_bench\\change_b_grad\\'
base_directory_g = 'Z:\\jenseno-opm\\Data\\2023_08_11_bench\\copy_change_b_grad_asc\\'
# base_directory_g = 'Z:\\Data\\2023_08_14_bench\\change_b_grad_100nT\\'
# base_directory_g = 'Z:\\Data\\2023_08_15_bench\\grad_change_b_100nT\\'
# base_directory_g = 'Z:\\jenseno-opm\\Data\\2023_08_15_bench\\grad_change_b_0nT_80Hz\\'

subfolder_list_g = os.listdir(base_directory_g)

# base_directory_m =  'Z:\\Data\\2023_08_11_bench\\change_b_mag\\'
base_directory_m = 'Z:\\jenseno-opm\\Data\\2023_08_11_bench\\copy_change_b_mag_asc\\'
# base_directory_m = 'Z:\\Data\\2023_08_14_bench\\change_b_mag_100nT\\'
# base_directory_m = 'Z:\\Data\\2023_08_15_bench\\mag_change_b_100nT\\'
# base_directory_m = 'Z:\\jenseno-opm\\Data\\2023_08_15_bench\\mag_change_b_0nT_80Hz\\'

subfolder_list_m = os.listdir(base_directory_m)

print('Loading.....')

Data_list_g = list()
for cur_subfolder in subfolder_list_g:
    print('...')
    Data_list_g.append(obs.Joined(base_directory_g, cur_subfolder,Gain, T1, T2))
    print('Loaded ' + cur_subfolder)
    
    
Data_list_m = list()
for cur_subfolder in subfolder_list_m:
    print('...')
    Data_list_m.append(obs.Joined(base_directory_m, cur_subfolder,Gain, T1, T2))
    print('Loaded ' + cur_subfolder)



#%% 
if runtype ==1:
    Vin = np.array([3.23,4.23,5.23,6.23,7.23,8.23,9.23,10.23,11.23,12.23])

if runtype ==2:
    Vin = np.array([3.23,3.23,4.23,4.23,5.23,5.23,6.23,6.23,7.23,7.23,8.23,8.23,9.23,9.23,10.23,10.23,11.23,11.23,12.23,12.23])

b_field = Vin/6.45 #6.45 mB/nT

freqs = 8

maxvals_all_g = list()
maxvals_all_m = list()
atten_avg = list()
atten_err= list()

att_a_l = list()
att_b_l = list()

a_err = list()
b_err = list()


for i in range(len(b_field)):
    if runtype ==1:
        maxvals_all_g = Data_list_g[i].PiD.findlmax(freqs)
        maxvals_all_m = Data_list_m[i].PiD.findlmax(freqs)
        
        maxvals_all_h_g = Data_list_g[i].PiD.findlmax(freqs*2)
        maxvals_all_h_m = Data_list_g[i].PiD.findlmax(freqs*2)
        
        atten = (np.mean(maxvals_all_m)+np.mean(maxvals_all_h_m))-(np.mean(maxvals_all_g)+np.mean(maxvals_all_h_g))
        
        
        atten_std = ((np.std(maxvals_all_m)**2 + np.std(maxvals_all_h_m)**2) + 
                     (np.std(maxvals_all_g)**2+np.std(maxvals_all_h_g)**2))**0.5
    
        atten_avg.append(atten)
        atten_err.append(atten_std) #not right
        
    elif runtype ==2:
        if i%2==0:
            
            
            mv_g_a = Data_list_g[i].PiD.findlmax(freqs)
            mv_g_b = Data_list_g[i+1].PiD.findlmax(freqs)
            
            mv_m_a = Data_list_m[i].PiD.findlmax(freqs)
            mv_m_b = Data_list_m[i+1].PiD.findlmax(freqs)
            
            att_a = np.mean(mv_m_a)-np.mean(mv_g_a)
            att_b = np.mean(mv_m_b)-np.mean(mv_g_b)
            
            att_a_s = (np.std(mv_m_a)**2+np.std(mv_g_a)**2)**0.5
            att_b_s = (np.std(mv_m_b)**2+np.std(mv_g_b)**2)**0.5
            
            att_a_l.append(att_a)
            att_b_l.append(att_b)
            
            a_err.append(att_a_s)
            b_err.append(att_b_s)
        
        else:
            continue

# atten_std = ex_B/np.array(maxvals_std)

V_plot = np.array([3.23,4.23,5.23,6.23,7.23,8.23,9.23,10.23,11.23,12.23])

b_plot = V_plot/6.45

fig, ax = plt.subplots()
ax.grid()
ax.set_title('Attenuation Factor of 8Hz sine wave at different fields (Mag/Grad)')
ax.set_xlabel('Applied field (nT)')
ax.set_ylabel('Attenuation Factor (dB)')
ax.set_xlim(0.25,2)

for j in range(len(atten_avg)):
    
    ax.errorbar(b_plot[j],atten_avg[j],fmt='k.',yerr=atten_err[j])
    
    

# plt.figure()
# plt.errorbar(b_plot,att_a_l,yerr=a_err,fmt='bo')
# plt.errorbar(b_plot,att_b_l,yerr=b_err,fmt='go')


#%% plot spectra
for i in range(len(b_field)):

    fig, ax = plt.subplots()
    Data_list_g[i].plotavgpower(fig,ax)
    Data_list_m[i].plotavgpower(fig,ax)

#%%

# os.chdir('Z:\\Quantum-Neuroscience_CHBH\\HarryRepo\\Python\\Analysis\\REAL_PAPER_ANALYSIS\\Export_data\\')

# CSV_filename = 'Atten_vs_b_0nT_80Hz_1.csv'
# CSV_header = ['Field (nT)', 'Atten','Atten_err']
# # CSV_array = np.transpose([freqs_plot,atten_avg,atten_err])
# CSV_array = np.transpose([b_plot,att_b_l,b_err])

# obs.ToCSV(CSV_array,CSV_header,CSV_filename)






