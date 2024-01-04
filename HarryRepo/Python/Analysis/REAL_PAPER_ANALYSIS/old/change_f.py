# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:42:06 2023

@author: H
"""
# from Proc import obs
import obs
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

runtype = 1 #1 is a single set of data,2 is asc/desc runs

# base_directory_g = 'Z:\\Data\\2023_08_10_bench\\change_f_grad_0_b1nT\\'
# base_directory_g = 'Z:\\Data\\2023_08_11_bench\\change_f_grad\\'
# base_directory_g = 'Z:\\Data\\2023_08_10_bench\\change_f_100_grad\\'
# base_directory_g = 'Z:\\Data\\2023_08_14_bench\\change_f_grad_0nT_b0.5nT\\'
base_directory_g = 'Z:\\jenseno-opm\\Data\\2023_08_15_bench\\grad_change_f_100nT\\'


subfolder_list_g = os.listdir(base_directory_g)

# base_directory_m =  'Z:\\Data\\2023_08_10_bench\\change_f_mag_0_1bT\\'
# base_directory_m = 'Z:\\Data\\2023_08_11_bench\\change_f_mag\\'
# base_directory_m = 'Z:\\Data\\2023_08_10_bench\\change_f_100_mag\\'
# base_directory_m = 'Z:\\Data\\2023_08_14_bench\\change_f_mag_0nT_b0.5nT\\'
base_directory_m = 'Z:\\jenseno-opm\\Data\\2023_08_15_bench\\mag_change_f_100nT\\'


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
    Data_list_m.append(obs.Joined(base_directory_m, cur_subfolder,Gain, T1, T2))
    print('Loaded ' + cur_subfolder)



#%% 

if runtype ==1:
    freqs = [5,10,15,20,25,30,35,40,45,55,60,65,70,75,80,85,90,95]
elif runtype ==2:
    freqs = [5,5,10,10,15,15,20,20,25,25,30,30,35,35,40,40,45,45,
     55,55,60,60,65,65,70,70,75,75,80,80,85,85,90,90,95,95]


atten_avg = list()
atten_err = list()

att_a_l = list()
att_b_l = list()

a_err = list()
b_err = list()

for i in range(len(freqs)):
    if runtype ==1:
        maxvals_all_g = Data_list_g[i].PiD.findlmax(freqs[i])
        maxvals_all_m = Data_list_m[i].PiD.findlmax(freqs[i])
        
        maxvals_all_h_g = Data_list_g[i].PiD.findlmax(freqs[i]*2)
        maxvals_all_h_m = Data_list_g[i].PiD.findlmax(freqs[i]*2)
        
        atten = (np.mean(maxvals_all_m)+np.mean(maxvals_all_h_m))-(np.mean(maxvals_all_g)+np.mean(maxvals_all_h_g))
        
        
        atten_std = ((np.std(maxvals_all_m)**2 + np.std(maxvals_all_h_m)**2) + 
                     (np.std(maxvals_all_g)**2+np.std(maxvals_all_h_g)**2))**0.5
    
        atten_avg.append(atten)
        atten_err.append(atten_std) #not right
        
    elif runtype ==2:
        if i%2==0:
            
            
            maxvals_all_g = np.concatenate((Data_list_g[i].PiD.findlmax(freqs[i]),Data_list_g[i+1].PiD.findlmax(freqs[i+1])))
            maxvals_all_m = np.concatenate((Data_list_m[i].PiD.findlmax(freqs[i]),Data_list_m[i+1].PiD.findlmax(freqs[i+1])))
        
            atten = np.mean(maxvals_all_m)-np.mean(maxvals_all_g)
            atten_std = (np.std(maxvals_all_m)**2+np.std(maxvals_all_g)**2)**0.5
        
            atten_avg.append(atten)
            atten_err.append(atten_std) #not right
            
            mv_g_a = Data_list_g[i].PiD.findlmax(freqs[i])
            mv_g_b = Data_list_g[i+1].PiD.findlmax(freqs[i+1])
            
            mv_m_a = Data_list_m[i].PiD.findlmax(freqs[i])
            mv_m_b = Data_list_m[i+1].PiD.findlmax(freqs[i+1])
            
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


fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel('Frequency')
ax.set_ylabel('Attenuation (dB)')


freqs_plot = [5,10,15,20,25,30,35,40,45,55,60,65,70,75,80,85,90,95]
for j in range(len(atten_avg)):
    
    ax.errorbar(freqs_plot[j],atten_avg[j],fmt='k.',yerr=atten_err[j])

# if runtype ==2:
#     plt.errorbar(freqs_plot,att_a_l,yerr=a_err,fmt='bo')
#     plt.errorbar(freqs_plot,att_b_l,yerr=b_err,fmt='go')


#%% 
# os.chdir('Z:\\Quantum-Neuroscience_CHBH\\HarryRepo\\Python\\Analysis\\REAL_PAPER_ANALYSIS\\Export_data\\')

# CSV_filename = 'Atten_vs_f_0nT_1.csv'
# CSV_header = ['Freqs (Hz)', 'Atten','Atten_err']
# # CSV_array = np.transpose([freqs_plot,atten_avg,atten_err])
# CSV_array = np.transpose([freqs_plot,att_a_l,a_err])

# obs.ToCSV(CSV_array,CSV_header,CSV_filename)



# xscale= 'log'

# for i in range(len(freqs)):
#     title1 = 'GRAD: '+ str(round(freqs[i],1)) + 'uW Laser Power'
#     Data_list_g[i].Noise_spectrum_title(title1,xscale)
#     # title2 = 'MAG: '+ str(round(l_power[i],1)) + 'uW Laser Power'
#     # Data_list_m[i].Noise_spectrum_title(title2,xscale)
    
    

# for i in range(len(freqs)):

#     fig, ax = plt.subplots()
#     Data_list_g[i].plotavgpower(fig,ax)
#     Data_list_m[i].plotavgpower(fig,ax)


grad_fx = Data_list_g[0].PiD.xf_flat_q
    
grad_fy = np.squeeze(Data_list_g[0].PiD.yf_flat_q)


fig4,ax4 = plt.subplots()

ax4.plot(grad_fx[:700],grad_fy[:700],'g',label='grad')
ax4.set_title('Gradiometer spectrum over 7s')
ax4.set_ylabel('Signal Amplitude (T)')
ax4.set_xlabel('Frequency (Hz) ')
ax4.set_yscale('log')
ax4.set_xscale('linear')
ax4.grid(True)

