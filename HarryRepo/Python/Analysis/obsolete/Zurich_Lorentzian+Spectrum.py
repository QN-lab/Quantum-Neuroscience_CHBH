# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:22:49 2022
Adapted Yulia's code
@author: Harry
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import pandas as pd
import math
from scipy.optimize import curve_fit
import statistics as stats 
import re


#TO FIX: 
    #LEGENDS MATCH BUT ARE IN A STRANGE ORDER, REORDER THEM CHRONOLOGICALLY
    #query input for type of noise floor calculation
#%% preamble & User inputs
csv_sep = ';' #separator for saved CSV
sfreq = 2100 #frequency at which to start the fitting
Cell_num = 2 #Cell we interrogate

#%% Data Read-in

#Resonance Data
Folderpath_r = 'Z:/Data/2022_12_1/C{}_res_power_000/'.format(Cell_num)
Filename_r = 'dev3994_demods_0_sample_00000.csv'
Headername_r = 'dev3994_demods_0_sample_header_00000.csv'

resonance = pd.read_csv (Folderpath_r+Filename_r,sep=csv_sep)    
res_legends = pd.read_csv (Folderpath_r+Headername_r,sep=csv_sep)

#Spectrum Data with filter
Folderpath_s = 'Z:/Data/2022_12_1/C{}_spectrum_power_000/'.format(Cell_num)
Filename_spectrum = 'dev3994_demods_0_sample_xiy_fft_abs_avg_00000.csv'
Filename_filter = 'dev3994_demods_0_sample_xiy_fft_abs_filter_00000.csv'
Headername_s = 'dev3994_demods_0_sample_xiy_fft_abs_avg_header_00000.csv'

spect = pd.read_csv (Folderpath_s+Filename_spectrum,sep=csv_sep)
spect_filter = pd.read_csv (Folderpath_s+Filename_filter,sep=csv_sep)
spect_legends = pd.read_csv (Folderpath_s+Headername_s,sep=csv_sep)

#Load in desired data, not sure why removing the last datapoint
ind_x = resonance.index[resonance['fieldname'] == 'x'].tolist()
x_data=resonance.iloc[ind_x,4:-1]
ind_y = resonance.index[resonance['fieldname'] == 'y'].tolist()
y_data=resonance.iloc[ind_y,4:-1]
ind_ph = resonance.index[resonance['fieldname'] == 'phase'].tolist()
ph_data=resonance.iloc[ind_ph,4:-1]
ind_frq = resonance.index[resonance['fieldname'] == 'frequency'].tolist()
frq_data=resonance.iloc[ind_frq,4:-1]

#Load Headers for each run
legends_res=res_legends['history_name'].tolist()
legends_spectr=spect_legends['history_name'].tolist()

#Load power array from headernames
regex = '{} (.*\d)u'.format(Cell_num)

def pull_header_power(header):
    patch = np.array(range(len(header)))
    Pow = np.zeros([patch.shape[0]])
    for i in patch:
        out = re.findall(regex, header[i])
        Pow[i] = float(out[0])
    return Pow, patch

#If both resonance and spectrum data don't exist for a certain power, 
    #this will select only the data for analysis where both datasets exist

Pow1, patch1 = pull_header_power(legends_res)
Pow2, patch2 = pull_header_power(legends_spectr)

if patch1.shape[0] != patch2.shape[0]:
    print('##################################################################')
    print('WARNING: data misalignment, ignoring non-matched data')
    if patch1.shape[0] > patch2.shape[0]:
        diff = patch1.shape[0] - patch2.shape[0]
        indx = np.zeros([patch2.shape[0]])
        for i in patch2:
            ind_i = np.where(Pow2[i]==Pow1)
            indx[i] = ind_i[0]
        Pow = Pow1[indx.astype(int)]
    elif patch1.shape[0] < patch2.shape[0]:
        diff = patch2.shape[0] - patch1.shape[0]
        indx = np.zeros([patch1.shape[0]])
        for i in patch1:
            ind_i = np.where(Pow1[i]==Pow2)
            indx[i] = ind_i[0]
        Pow = Pow2[indx.astype(int)]
    print('Running Data for {} uW runs, as both both files exist for these runs'.format(Pow))
    print('##################################################################')
elif patch1.shape[0] == patch2.shape[0]:
    print('All runs have both Spectrum and Resonance Data')
    Pow = Pow1
patch = np.array(range(len(Pow)))

#Save as single Data Array
data=np.zeros((patch.shape[0],frq_data.shape[1],4))
for x in patch:
    data[x,:,0]=np.array(frq_data.iloc[x,:])
    data[x,:,1]=np.array(x_data.iloc[x,:])
    data[x,:,2]=np.array(y_data.iloc[x,:])
    data[x,:,3]=np.array(ph_data.iloc[x,:])

#%% Fitting
plt.ion()
fig1 = plt.figure('fig1')
ax1 = plt.subplot(211)
#Plot Target data
for i in patch:
    #str=legends_res[i].replace('_',' ')
    plt.plot(data[i,:,0], data[i,:,1]*1000,label=str)
    plt.ylabel("quadrature, mV")
    #plt.legend([i for i in range(frq_data.shape[0])])
    #plt.legend(loc = 'upper right', fontsize = 8,labelspacing = 0.2)
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    
#Automatically create central frequency guess after a certain frequency(ignore zero field)
start_indx = np.abs(data[0,:,0] - sfreq).argmin()
peakindx = start_indx + data[:,start_indx:,1].argmax(axis=1) #index at which the lorentzian peak exists for each run

def _Lorentzian(x, amp, cen, wid, slope, offset):
    return (amp*(wid)**2/((x-cen)**2+(wid)**2)) + slope*x + offset 

k=0;
param_res=np.zeros((patch.shape[0],5))
fit_cov_mat= np.zeros((patch.shape[0],5,5))
field_res=np.zeros((patch.shape[0],5))
for j in range(patch.shape[0]):
    data_red=[]
    k=0
    for i in range(0,data[3,:,1].size-10,10):
        if abs(np.mean(data[patch[j],i+5:i+10,1])-np.mean(data[patch[j],i:i+5,1]))>5e-7:
            if k==0: data_red=np.asarray(data[patch[j],i:i+10,:]) 
            else: data_red=np.append(data_red,data[patch[j],i:i+10,:],axis=0)
            k+=1
    popt_lor, pcov_lor = curve_fit(_Lorentzian,data_red[:,0],data_red[:,1]*1000,p0=[1,data[j,peakindx[j],0],200,0,0]) #guess peak: min-max
    fit_cov_mat[j,:,:] = pcov_lor
    param_res[j,:]=popt_lor
    field_res[j]=71*1e-6*param_res[j,1]
#print(param_res) #(amplitude, central frequency, half width, slope, constant offset)

#Plot Fit
ax2 = plt.subplot(212, sharey = ax1)
for i in patch:
    plt.plot(data[i,:,0],_Lorentzian(data[i,:,0], *param_res[i,:]))
    plt.xlabel("frequency, Hz")
    plt.ylabel("Lorentzian Fit")
    plt.grid(color='k', linestyle='-', linewidth=0.5)


#Acquire Fit Errors for each run
fiterr = np.zeros((patch.shape[0],5))

for i in range(patch.shape[0]):
    fiterr[i,:] = np.sqrt(np.diag(fit_cov_mat[i,:,:]))


#%% SNR from spectrum

spectr_chunk_size=spect_legends['chunk_size'].tolist()
data_spec=np.zeros((spectr_chunk_size[1],len(legends_spectr)))
data_spec_tmp=np.zeros((spectr_chunk_size[1],len(legends_spectr)))
data_filter=np.zeros((spectr_chunk_size[1],len(legends_spectr)))
Spect_pick = np.zeros(len(legends_spectr))
Spect_noise_floor =np.zeros(len(legends_spectr))
med_spec = np.zeros((patch.shape[0],spectr_chunk_size[0]))
SNr=np.zeros(len(legends_spectr))

#query = input("Noise level estimation. 1-Median, 2-Manual floor:     ")


# interval to estimate noise floor
f1=-300
f2=-260

for i in range(len(legends_spectr)):
    if i==0: a=0 
    else: a+=spectr_chunk_size[i-1]   
    data_spec_tmp[:,i]=spect.iloc[a:a+spectr_chunk_size[i],2]
    data_filter[:,i]=spect_filter.iloc[a:a+spectr_chunk_size[i],2]
    data_spec[:,i]=data_spec_tmp[:,i]/data_filter[:,i]
    Spect_pick[i]=max(data_spec[:,i])

    med_spec[i] = stats.median(data_spec[:,i])*np.ones(spectr_chunk_size[0])
    SNr[i]=Spect_pick[i]/med_spec[i,0]
    
    # Spect_noise_floor[i]=stats.mean(data_spec[f1:f2,i])
    # SNr2[i]=Spect_pick[i]/Spect_noise_floor[i]
    
    
#Spectrum offsets, pulled from the first run, so if runs are significantly different, this needs to be changed.
pull_rel_off = spect_legends['grid_col_offset'].tolist()
frq_domain = np.linspace(pull_rel_off[0],-pull_rel_off[0],spectr_chunk_size[0])

#Plot Spectrum
fig2 = plt.figure('fig2')
for i in range(data_spec.shape[1]):
    str=legends_spectr[i].replace('_',' ')
    plt.semilogy(frq_domain,data_spec[:,i],label=str)
    plt.legend([i for i in range(frq_data.shape[0])])
    plt.xlabel("frequency, Hz")
    plt.ylabel("quadrature, mV")
    plt.legend(loc = 'best', fontsize = 8, labelspacing = 0.2)
    plt.grid(color='k', linestyle='-', linewidth=0.5)

fig21 = plt.figure('fig21')

for i in range(data_spec.shape[1]):
    plt.subplot(data_spec.shape[1],2,i+1)
    plt.semilogy(frq_domain,data_spec[:,i])
    plt.semilogy(frq_domain,med_spec[i,:],lw=2)
    plt.axis('off')

#%% Sensitivity

sensitivity=np.zeros(patch.shape[0])
width=np.zeros(patch.shape[0])
g=1/2
hbar=1.05e-34
mu=9.27e-24
width=2*abs(param_res[:,2])
sensitivity=(2*math.pi*width*hbar)/(g*mu*SNr) #was SNR[0:1] here but not sure why

#%% Power vs Params

amp_er = fiterr[:,0]
wid_er = 2*fiterr[:,2] #2: error prop because output is half width
centr_er = fiterr[:,1]

fig3 = plt.figure('fig3')

#Pow vs Width
plt.subplot(3,1,1)
plt.errorbar(Pow,width,yerr=wid_er,fmt='.b')
plt.ylabel('Width (Hz)')

#Pow vs Amplitude
plt.subplot(3,1,2)
plt.errorbar(Pow,param_res[:,0],yerr=amp_er,fmt='.b')
plt.ylabel('Fit Amplitude (mV)')

#Pow vs Central Frequency
plt.subplot(3,1,3)
plt.errorbar(Pow,param_res[:,1],yerr=centr_er,fmt='.b')
plt.ylabel('Central Frequency(Hz)')
plt.xlabel('Laser Power (mW)')

#Width/height Ratio
slope = param_res[:,0]/width
    #Add errors in quadrature
slope_er= slope*np.sqrt(((wid_er/width)**2)+((amp_er/param_res[:,0])**2))

Fig4 = plt.figure('fig4')
#plt.plot(Pow,param_res[:,0]/width,'bD')
plt.errorbar(Pow,slope,yerr=slope_er,fmt='.b')
plt.ylabel('Amplitude over width (mV/Hz)')
plt.xlabel('Laser Power (mW)')
plt.grid(color='k', linestyle='-', linewidth=0.5)

#Sensitivity
    #Error
sens_er = (2*math.pi*wid_er*hbar)/(g*mu*SNr) # assumes no error on SNr, probably need to account for this
Fig5 = plt.figure('fig5')
plt.errorbar(Pow,sensitivity,yerr=sens_er,fmt='.b')
plt.ylabel('Sensitivity')
plt.xlabel('Laser Power (mW)')
plt.grid(color='k', linestyle='-', linewidth=0.5)

