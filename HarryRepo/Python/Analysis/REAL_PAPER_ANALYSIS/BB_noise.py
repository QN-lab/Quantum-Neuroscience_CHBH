# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:42:06 2023

@author: H
"""
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

###############################################################################
#%% Resonance

# r_sig = 'dev3994_demods_0_sample_00000.csv'
# r_header = 'dev3994_demods_0_sample_header_00000.csv'

# res_folder_g = 'Z:\\jenseno-opm\\Data\\2023_08_18_bench\\grad_noise_high_res\\grad_noise_res_000\\'
# # res_folder_g = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\grad_noise_res\\grad_noise_res_000\\'

# res_g, res_legends_g = HCA.ReadData(res_folder_g,r_sig,r_header,';')
# resonance_g = obs.Resonance(res_g, res_legends_g, 1100)
# # resonance_g.plot_both()

# res_folder_m = 'Z:\\jenseno-opm\\Data\\2023_08_18_bench\\mag_noise_high_res\\mag_noise_res_000\\'
# # res_folder_m = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\mag_noise_res\\mag_noise_res_000\\'
# res_m, res_legends_m = HCA.ReadData(res_folder_m,r_sig,r_header,';')
# resonance_m = obs.Resonance(res_m, res_legends_m, 1100)
# # resonance_m.plot_Y()

# fig,ax = plt.subplots()

# resonance_g.plot_both_1run(fig,ax)
# ax.set_xlim((1050,1850))

#%% 
Gain = 14.2
T1=0
T2=3

# base_directory_g = 'Z:\\Data\\2023_08_18_bench\\grad_noise_high\\'
base_directory_g = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\grad_noise\\'
# base_directory_g = 'Z:\\Data\\2023_08_25_bench\\grad_100nT_bb_noise\\'
subfolder_list_g = os.listdir(base_directory_g)

# base_directory_m = 'Z:\\Data\\2023_08_18_bench\\mag_noise_high\\'
base_directory_m = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\mag_noise\\'
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

# mV = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65])
mV = np.array([0,3,6,9,12,15,18,21,24,27,30,33])
# mV = np.array([3,6,9,12,15,18,21,24,27,30,33])

b_field = mV/6.45

#Attenuation spectrum!!!

atten_l = list()

for i in range(len(b_field)):
    
    atten = np.subtract(Data_list_m[i].PiD.yf_avg,Data_list_g[i].PiD.yf_avg)
    atten_l.append(atten)


# for i in range(len(b_field)):
#     plt.figure()
#     plt.plot(Data_list_g[i].PiD.xf,atten_l[i])
#     plt.xlim([0,100])
#     plt.ylim([0,50])
#     title = 'Noise Amp: '+ str(round(b_field[i],2)) + 'nT'
#     plt.title(title)

plt.figure()
for i in range(len(b_field)):
    label = str(round(b_field[i],1)) + 'nT'
    plt.plot(Data_list_g[i].PiD.xf,atten_l[i],label=label)
    # plt.xlim([0,100])
    plt.ylim([0,50])
    plt.legend()
    
# nc=round(len(b_field)*2)
# fig1,ax1 = plt.subplots()
# cm = plt.get_cmap('inferno')
# ax1.set_prop_cycle(color=[cm(1.*i/nc) for i in range(nc)])
# ax1.set_xlim([0,100])
# ax1.set_ylim([0,50])
# ax1.set_ylabel('Attenuation (dB)')
# ax1.set_xlabel('Frequency (Hz)')
# # ax1.set_title('Attenuation of 100Hz broadband noise, averaged over trials')

# for i in range(len(b_field)):
#     label = str(round(b_field[i],1)) + 'nT'
#     smooth_y_i = np.array(atten_l[i][:250]).reshape(-1, 5).mean(axis=1) #250points goes up to ~100Hz
#     smooth_y = np.delete(smooth_y_i,[3,4])                                  #index 3 is the 6Hz sine which we don't want to see/
#     smooth_x_i = np.linspace(0, 100, num=len(smooth_y_i), endpoint=True)
#     smooth_x = np.delete(smooth_x_i,[3,4])
#     # if i >=1:
#     ax1.plot(smooth_x,smooth_y,label=label)
#     ax1.legend(title='Amplitude of \n applied noise',bbox_to_anchor=(1.25, 1.05))
#     plt.grid(True)
#     # elif i ==0:
#         # pass




# #%%

# for i in range(len(b_field)):

#     fig, ax = plt.subplots()
#     Data_list_g[i].plot_x_power(fig,ax,'g')
    
#     Data_list_m[i].plot_x_power(fig,ax,'b')

#%% Senstivity

# sens_g = np.zeros((2,len(b_field)))
# sens_m = np.zeros((2,len(b_field)))

# for i in range(len(b_field)):
#     sens_g[:,i] = obs.sens_SNR(Data_list_g[i],resonance_g)
#     sens_m[:,i] = obs.sens_SNR(Data_list_m[i],resonance_m)
    
    
# plt.figure()
# plt.plot(b_field,sens_g/1e-15,'g.',label='Grad')
# plt.plot(b_field,sens_m/1e-15,'b.',label='Mag')
# plt.ylabel('sensitivity(fT/rHz)')
# plt.xlabel('Applied BB noise (nT)')
# plt.xlim([-1,6])
# plt.grid(True)
# plt.yscale('log')
# plt.legend()
    
# g_sens = list()
# m_sens = list()

# for i in range(len(b_field)):
#     g_sens.append(obs.sens_std(Data_list_g[i]))
#     m_sens.append(obs.sens_std(Data_list_m[i]))


# fig,ax = plt.subplots()
# ax.set_yscale('log')
# ax.grid(True)
# ax.set_title('Sensitivity as std.dev. over 1s')
# ax.set_xlabel('Applied Field (nT)')
# ax.set_ylabel('Sensitivity (T)')



# for i in range(len(b_field)):
    
#     #ON
#     ax.errorbar(b_field[i],g_sens[i][0][0],yerr=g_sens[i][0][1],fmt = 'g+')
#     ax.errorbar(b_field[i],m_sens[i][0][0],yerr=m_sens[i][0][1],fmt = 'b+')
    
#     ax.errorbar(b_field[i],g_sens[i][1][0],yerr=g_sens[i][1][1],fmt = 'g.')
#     ax.errorbar(b_field[i],m_sens[i][1][0],yerr=m_sens[i][1][1],fmt = 'b.')
    
    


#%% Plotting Field

fig,ax = plt.subplots(constrained_layout=True)
fig.set_figwidth(12)
ax.set_yscale('log')
ax.set_ylabel('Noise Level ($T \: Hz^{-1/2}$)',fontsize=15)
ax.set_xlabel('Frequency ($Hz$)',fontsize=15)
for i in range(len(b_field)):
    label = str(round(b_field[i],1)) + 'nT'
    ax.plot(Data_list_g[i].PiD.xf,Data_list_m[i].PiD.yf_avg_a,label=label)
    ax.set_xlim([0,100])
    ax.legend()
   
    
fig,ax = plt.subplots(constrained_layout=True)
fig.set_figwidth(12)
ax.set_yscale('log')
ax.set_ylabel('Noise Level ($T \: Hz^{-1/2}$)',fontsize=15)
ax.set_xlabel('Frequency ($Hz$)',fontsize=15)
for i in range(len(b_field)):
    label = str(round(b_field[i],1)) + 'nT'
    ax.plot(Data_list_g[i].PiD.xf,Data_list_g[i].PiD.yf_avg_a,label=label)
    ax.set_xlim([0,100])
    ax.legend()
    

#%% Export

ydata_i = np.vstack(atten_l)
xdata_i = Data_list_g[i].PiD.xf

# rminx = [0,17,18,19,149,150,151]
    
# ydata_i = np.zeros((12,31))
# xdata_i = np.zeros((12,251))
# for i in range(len(b_field)):

        
#     ydata_i[i,:] = np.array(atten_l[i]).reshape(-1, 5).mean(axis=1)
#     xdata_i = np.array(Data_list_g[i].PiD.xf).reshape(-1, 5).mean(axis=1)

ydata = ydata_i
xdata = xdata_i

# os.chdir('Z:\\Publications_submission\\Gradiometer Paper Prep\\Origin files\\Harry_data\\')
    
# title = 'Atten_vs_BB_noise_100nT'

# colnames = list()

# for i in b_field:
#     colnames.append(str(round(i,1)))

# df = pd.DataFrame(ydata,colnames)
# df = df.T
# df['Freq'] = xdata

# df.to_csv(title+'.csv')

#Test exporting correct data
plt.figure()
plt.xlim(0,100)
plt.ylim(0,40)
for i in range(ydata.shape[0]):
    plt.plot(xdata,ydata[i,:])
    
    
    
    