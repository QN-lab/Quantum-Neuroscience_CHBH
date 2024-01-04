# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 17:42:06 2023

@author: H
"""
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

base_directory_g = 'Z:\\Data\\2023_08_10_bench\\change_b_grad\\'

subfolder_list_g = os.listdir(base_directory_g)

base_directory_m =  'Z:\\Data\\2023_08_10_bench\\change_b_mag\\'

subfolder_list_m = os.listdir(base_directory_m)

headers = ['dev3994_demods_0_sample_auxin1_avg_header_00000.csv',
                 'dev3994_demods_0_sample_trigin2_avg_header_00000.csv',
                 'dev3994_demods_0_sample_xiy_fft_abs_avg_header_00000.csv',
                 'dev3994_pids_0_stream_shift_avg_header_00000.csv'
                 ]

sigs = ['dev3994_demods_0_sample_auxin1_avg_00000.csv',             #Arduino
              'dev3994_demods_0_sample_trigin2_avg_00000.csv',      #Trigger
              'dev3994_demods_0_sample_xiy_fft_abs_avg_00000.csv',  #XiY
              'dev3994_pids_0_stream_shift_avg_00000.csv'           #PID Signal
              ]

###############################################################################
#%%
def ReadData(cur_fold,sigs,headers):
    out_sig = list()
    out_headers = list()
    for i in range(len(sigs)):
        out_sig.append(pd.read_csv(cur_fold+sigs[i],sep=';'))
        out_headers.append(pd.read_csv(cur_fold+headers[i],sep=';'))
    return out_headers, out_sig
 
#Loading functions
def pull_headers(header_df):
    chunk_size = header_df['chunk_size'].tolist()
    chunk_num = header_df['chunk_number'].tolist()
    run_name = header_df['history_name'].tolist()
    return chunk_size, chunk_num, run_name

def pull_signals(sig_df):
    chunk = sig_df['chunk'].tolist()
    timestamps = np.array(sig_df['timestamp'].tolist())
    data = sig_df['value'].tolist() #THINK!!!!
    
    return chunk, timestamps, data
###############################################################################
#General Purpose pre-processing
def Janitor(dirty_chunked_data, trigger_chunked_data, patch):
    #Takes dirty data containing time-matched arduino triggers and removes them into a new variable. 
    
    cleaned_data = dirty_chunked_data
    
    for q in patch:
        if (trigger_chunked_data[q,:]>= 4.5).any(): 
            cleaned_data[q,:] = 0 #somehow changes the raw data to have 0 in these spots
            
    cleaned_data = cleaned_data[~np.all(cleaned_data == 0, axis=1)]
    
    return cleaned_data

###############################################################################
#Main Processing 

def Powerise(field_data,chunked_time,chunk):
    
    N = field_data.shape[1]
    T = chunked_time[0,1]-chunked_time[0,0]
    xf = fftfreq(N,T)[:N//2]
    
    yf_chunked = np.zeros((len(chunk),len(xf)))

    for a in chunk:
        
        yf = (2/N)*fft(field_data[a,:])
    
        yf_chunked[a,:] = 20*np.log10(np.abs(yf[0:N//2])) #Field?
    
    return xf, yf_chunked

###############################################################################
#%%
class Data_extract:
        
    def __init__(self, mounted_headers,mounted_sigs): 
        self.chunk_size, self.chunk_num, self.run_name = pull_headers(mounted_headers)
        self.chunk, self.timestamps, self.data = pull_signals(mounted_sigs)
        
        self.chunked_data = np.array(self.data).reshape(len(self.chunk_num),self.chunk_size[0])
        
        chunked_timestamps = self.timestamps.reshape(len(self.chunk_num),self.chunk_size[0])

        self.chunked_time = np.zeros(chunked_timestamps.shape)
        for i in range(len(self.chunk_num)):
            self.chunked_time[i,:] = (chunked_timestamps[i,:] - chunked_timestamps[i,0])/60e6



class PiD_processing(Data_extract):
    def __init__(self,mounted_headers,mounted_sigs,Trigger_obj):
        
        Data_extract.__init__(self, mounted_headers,mounted_sigs)
        
        #Cleaning
        self.clean_chunked_data = Janitor(self.chunked_data,Trigger_obj.chunked_data,self.chunk_num)
        self.Field = self.clean_chunked_data*Gain*0.071488e-9
        self.clean_chunks = list(range(len(self.clean_chunked_data)-1))
        
        #######################################################################
        #Spectrum processing 
        
        self.Roi_sidx = np.searchsorted(self.chunked_time[0,:], 0.0, side="left") #correct for 200ms trigger
        self.Roi_fidx = np.searchsorted(self.chunked_time[0,:], 3, side="left")
        
        self.RoI = self.Field[:,self.Roi_sidx:self.Roi_fidx]
        self.quiet_region = self.Field[:,self.Roi_fidx+1:] #CHANGE TO END AFTER 1 SECOND BECAUSE THE END OF TRIALS CONTAIN OSCILLATIONS
        
        self.xf, self.yf_chunked = Powerise(self.RoI,self.chunked_time,self.clean_chunks)
        
        self.yf_avg = np.mean(self.yf_chunked,axis=0)
        self.yf_std = np.std(self.yf_chunked,axis=0)
        
    def findlmax(self,freq):
        
        idx = (np.abs(self.xf-freq)).argmin()
        maxval = np.zeros(len(self.clean_chunks))
    
        for a in self.clean_chunks:
        
            roi = self.yf_chunked[a,int(idx-2):int(idx+2)]
            maxval[a] = np.max(roi)
        
        return maxval
        
###############################################################################
#%% 

#Read in data 
class Joined:
    def __init__(self,base_directory,subfolder,headers,sigs):
        
        cur_fold = base_directory+subfolder+'\\'
        mounted_headers, mounted_sigs = ReadData(cur_fold,sigs,headers)
        
        self.subfolder_name = subfolder
        self.Arduino = Data_extract(mounted_headers[0],mounted_sigs[0])
        self.Trigger_in = Data_extract(mounted_headers[1],mounted_sigs[1])
        self.XiY = Data_extract(mounted_headers[2],mounted_sigs[2])
        self.PiD = PiD_processing(mounted_headers[3],mounted_sigs[3],self.Arduino)

    def plotpower(self,fig,ax):

        ax.plot(self.PiD.xf,self.PiD.yf_chunked.shape)
        # ax.set_yscale('log')
        ax.set_xlim([0, 100])
        ax.set_title(self.subfolder_name)
    
    def plotavgpower(self,fig,ax):
       
        ax.plot(self.PiD.xf,self.PiD.yf_avg)
        # ax.set_yscale('log')
        ax.set_xlim([0, 100])
        ax.set_title(self.subfolder_name)
        
print('Loading.....')

Data_list_g = list()
for cur_subfolder in subfolder_list_g:
    print('...')
    Data_list_g.append(Joined(base_directory_g, cur_subfolder, headers, sigs))
    print('Loaded ' + cur_subfolder)
    
    
Data_list_m = list()
for cur_subfolder in subfolder_list_m:
    print('...')
    Data_list_m.append(Joined(base_directory_m, cur_subfolder, headers, sigs))
    print('Loaded ' + cur_subfolder)



#%% 

b_field = [0.5,0.5,0.75,0.75,1.0,1.0,1.25,1.25,1.5,1.5,1.75,1.75,2.0,2.0,2.25,2.25,2.50,2.50]

freqs = 10

maxvals_all_g = list()
maxvals_all_m = list()
atten_avg = list()
atten_err= list()

att_a_l = list()
att_b_l = list()


for i in range(len(b_field)):
    if i%2==0:
        
        maxvals_all_g = np.concatenate((Data_list_g[i].PiD.findlmax(freqs),Data_list_g[i+1].PiD.findlmax(freqs)))
        maxvals_all_m = np.concatenate((Data_list_m[i].PiD.findlmax(freqs),Data_list_m[i+1].PiD.findlmax(freqs)))

        atten = np.mean(maxvals_all_m)-np.mean(maxvals_all_g)
        atten_std = (np.std(maxvals_all_m)**2+np.std(maxvals_all_g)**2)**0.5
    
        atten_avg.append(atten)
        atten_err.append(atten_std) #not right
    
        
    
        mv_g_a = Data_list_g[i].PiD.findlmax(freqs)
        mv_g_b = Data_list_g[i+1].PiD.findlmax(freqs)
        
        mv_m_a = Data_list_m[i].PiD.findlmax(freqs)
        mv_m_b = Data_list_m[i+1].PiD.findlmax(freqs)
        
        att_a = np.mean(mv_m_a)-np.mean(mv_g_a)
        att_b = np.mean(mv_m_b)-np.mean(mv_g_b)
        
        att_a_l.append(att_a)
        att_b_l.append(att_b)
        
        
    else:
        continue

# atten_std = ex_B/np.array(maxvals_std)

b_plot = [0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.50]

fig, ax = plt.subplots()
ax.grid()
ax.set_title('Attenuation Factor of 10Hz sine wave at different fields (Mag/Grad)')
ax.set_xlabel('Applied field (nT)')
ax.set_ylabel('Attenuation Factor (dB)')
ax.set_xlim(0.25,2.75)

for j in range(len(atten_avg)):
    
    ax.errorbar(b_plot[j],atten_avg[j],fmt='b.',yerr=atten_err[j])
    
    


plt.plot(b_plot,att_a_l,'bo')
plt.plot(b_plot,att_b_l,'go')
plt.xlim(0.25,2.75)

#%% plot spectra

# for i in range(len(b_field)):

#     fig, ax = plt.subplots()
#     Data_list_g[i].plotavgpower(fig,ax)
#     Data_list_m[i].plotavgpower(fig,ax)


