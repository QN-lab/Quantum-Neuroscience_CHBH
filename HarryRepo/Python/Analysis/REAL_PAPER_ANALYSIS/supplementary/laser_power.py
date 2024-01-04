# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:42:06 2023

@author: H
"""
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
T1 = 0
T2 = 2

base_directory_g = 'Z:\\jenseno-opm\\Data\\2023_08_09_bench\\change_laser_grad\\'

subfolder_list_g = os.listdir(base_directory_g)

base_directory_m =  'Z:\\jenseno-opm\\Data\\2023_08_09_bench\\change_laser_mag\\'
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

l_power = [100,125,150,175,200,225,250,275,300]

freqs = 10

#CLASSIC 1-FREQ ATTENUATION
atten_avg = list()
atten_err = list()

att_a_l = list()
att_b_l = list()

a_err = list()
b_err = list()

for i in range(len(l_power)):
    maxvals_all_g = Data_list_g[i].PiD.findlmax(freqs)
    maxvals_all_m = Data_list_m[i].PiD.findlmax(freqs)
    
    maxvals_all_h_g = Data_list_g[i].PiD.findlmax(freqs*2)
    maxvals_all_h_m = Data_list_g[i].PiD.findlmax(freqs*2)
    
    atten = (np.mean(maxvals_all_m)+np.mean(maxvals_all_h_m))-(np.mean(maxvals_all_g)+np.mean(maxvals_all_h_g))
    
    
    atten_std = ((np.std(maxvals_all_m)**2 + np.std(maxvals_all_h_m)**2) + 
                 (np.std(maxvals_all_g)**2+np.std(maxvals_all_h_g)**2))**0.5

    atten_avg.append(atten)
    atten_err.append(atten_std) #not right

# # #SPECTRUM ATTENUATION
# atten_avg = list()
# atten_err = list()

# for i in range(len(l_power)):
    
#     atten = np.subtract(Data_list_m[i].PiD.yf_avg,Data_list_g[i].PiD.yf_avg)
#     # atten_e = 
#     atten_avg.append(np.mean(atten[:99]))
#     atten_err.append(np.std(atten[:99]))



g_sens = list()
m_sens = list()

for i in range(len(l_power)):
    g_sens.append(obs.sens_std(Data_list_g[i]))
    m_sens.append(obs.sens_std(Data_list_m[i]))

fig, ax = plt.subplots()
color1 = 'b'
color2 = 'r'
ax.grid()
# ax.set_title('Attenuationof 10Hz, 0.5nT amplitude sine wave')
ax.set_xlabel('Laser Power (uW)')
ax.set_ylabel('Attenuation Factor (dB)') #,color=color1
ax.set_xlim(80,320)
# ax.tick_params(axis='y', labelcolor=color1)
# ax2 = ax.twinx()
# ax2.set_ylabel('Sensitivity(pT/rHz)',color=color2)
# # ax2.set_ylim((0.85,1.35))
# ax2.tick_params(axis='y', labelcolor=color2)

for j in range(len(atten_avg)):
    
    ax.errorbar(l_power[j],atten_avg[j],fmt='b.', yerr=atten_err[j],xerr = 5)
    # ax2.errorbar(l_power[j],g_sens[j][0][0]/1e-12,yerr=g_sens[j][0][1]/1e-12,fmt = 'r.')
    
#attenuation on every value in order for the standard deviation to show properly
#%% Comparing widths

def ReadData2(folder_path,filename,headername,csv_sep):
    out_sig = pd.read_csv(folder_path+filename,sep=csv_sep)
    out_headers = pd.read_csv(folder_path+headername,sep=csv_sep)
    return out_sig, out_headers

def Res_read(folder_path,csv_sep):
    filename = 'dev3994_demods_0_sample_00000.csv'
    headername = 'dev3994_demods_0_sample_header_00000.csv'
    
    out_sig, out_headers = ReadData2(folder_path, filename, headername, csv_sep)
    
    return out_sig, out_headers

grad_res_folder = 'Z:\\jenseno-opm\\Data\\2023_08_09_bench\\change_laser_grad_res\\'
grad_res_subfolders = os.listdir(grad_res_folder)


mag_res_folder = 'Z:\\jenseno-opm\Data\\2023_08_09_bench\\change_laser_mag_res\\'
mag_res_subfolders = os.listdir(mag_res_folder)


grad_res = list()
for i in range(len(grad_res_subfolders)):

    qq1 = Res_read(grad_res_folder+grad_res_subfolders[i]+'\\',';')
    res1 = HCA.Resonance(qq1[0],qq1[1],1000)
    
    grad_res.append(res1)



mag_res = list()
for i in range(len(mag_res_subfolders)):

    qq2 = Res_read(mag_res_folder+mag_res_subfolders[i]+'\\',';')
    res2 = HCA.Resonance(qq2[0],qq2[1],1000)
    
    mag_res.append(res2)


# fig2, axs = plt.subplots(3,1,sharex = True)
# fig2.suptitle('Comparing mag and grad resonance shapes', fontsize=16)

# axs[0].set_xlim(80,320)
# axs[0].set_ylabel('width')

# axs[1].set_ylabel('amplitude')
# axs[2].set_ylabel('slope (A/W)')

# for i in range(len(l_power)):
#     axs[0].errorbar(l_power[i],mag_res[i].width,yerr =mag_res[i].width_err,fmt = 'b')
#     axs[0].errorbar(l_power[i],grad_res[i].width,yerr =grad_res[i].width_err,fmt = 'g')
    
#     axs[1].errorbar(l_power[i],mag_res[i].amplitude,yerr =mag_res[i].amplitude_err,fmt = 'b')
#     axs[1].errorbar(l_power[i],grad_res[i].amplitude,yerr =grad_res[i].amplitude_err,fmt = 'g')
    
#     axs[2].errorbar(l_power[i],mag_res[i].h_over_w,yerr =mag_res[i].h_over_w_err,fmt = 'b')
#     axs[2].errorbar(l_power[i],grad_res[i].h_over_w,yerr =grad_res[i].h_over_w_err,fmt = 'g')


# axs[0].legend(['mag','grad'],loc=4,fontsize="12")



fig2, axs = plt.subplots(2,1,sharex = True,constrained_layout=True)
# fig2.suptitle('Comparing mag and grad resonance shapes', fontsize=16)

axs[0].set_xlim(80,320)
axs[0].set_ylabel('Width (Hz)')
axs[0].grid(True)

axs[1].set_ylim(0,20)
axs[1].set_ylabel('Amplitude (uV)')
axs[1].set_xlabel('Laser Power b/f coupler (uW)')
axs[1].grid(True)

for i in range(len(l_power)):
    axs[0].plot(l_power[i],mag_res[i].width,'bX',markersize=10)
    axs[0].plot(l_power[i],grad_res[i].width,'gX',markersize=10)
    
    axs[1].plot(l_power[i],mag_res[i].amplitude,'bX',markersize=10)
    axs[1].plot(l_power[i],grad_res[i].amplitude,'gX',markersize=10)

axs[0].legend(['mag','grad'],loc=4,fontsize="12",bbox_to_anchor=(1, -0.3))



#%%

# xscale= 'log'

# for i in range(len(l_power)):
#     title1 = 'GRAD: '+ str(round(l_power[i],1)) + 'uW Laser Power'
#     Data_list_g[i].Noise_spectrum_title(title1,xscale)
#     # title2 = 'MAG: '+ str(round(l_power[i],1)) + 'uW Laser Power'
#     # Data_list_m[i].Noise_spectrum_title(title2,xscale)


#%%
    

# os.chdir('Z:\\jenseno-opm\\Publications_submission\\Gradiometer Paper Prep\\Origin files\\Harry_data\\')
    
# title = 'Atten_vs_Laser_power'
# d = {'laser_power': l_power,'Attenuation': atten_avg, 'Atten_err': atten_err
#      }

# df = pd.DataFrame(d)

# df.to_csv(title+'.csv')


