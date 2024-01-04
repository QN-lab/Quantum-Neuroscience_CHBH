# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:08:32 2023

@author: H
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('classic')
import math
# import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd

A_Param = [0,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4,4.5,5]
A_Param_r = A_Param[::-1]
csv_sep = ';'

updown = ['asc','des'] # 0nT asc,des; 60nT asc,des

looper_A = range(len(A_Param))

traces_m = list()
traces_g = list()

def ReadData(folder_path,filename,headername,csv_sep):
    out_sig = pd.read_csv(folder_path+filename,sep=csv_sep)
    out_headers = pd.read_csv(folder_path+headername,sep=csv_sep)
    return out_sig, out_headers

def DAQ_read_shift(folder_path,csv_sep):
    
    filename = 'dev3994_pids_0_stream_shift_avg_00000.csv'
    headername = 'dev3994_pids_0_stream_shift_avg_header_00000.csv'

    out_sig, out_headers = ReadData(folder_path, filename, headername, csv_sep)

    return out_sig, out_headers

class DAQ_Trigger:
    def __init__(self,sig,header):
        
        self.sig=sig.drop_duplicates(keep='last')
        self.header=header
        
        #Pull from headers
        self.ChunkSize = self.header['chunk_size'].tolist() #Size of each chunk
        self.patch = self.header['chunk_number'].tolist() # Prints a list of the number of runs
        self.run_names = self.header['history_name'].tolist()
        # self.filter_BW = self.header['history_name'].tolist()
        
        #Pull from Signals
        self.Chunk = self.sig['chunk'].tolist()
        self.timestamps = np.array(self.sig['timestamp'].tolist())
        self.Data = self.sig['value'].tolist()
        
        #Organise by Run
        self.chunked_data = np.array(self.Data).reshape(len(self.patch),self.ChunkSize[0])
        chunked_timestamps = self.timestamps.reshape(len(self.patch),self.ChunkSize[0])
        
        #WHEN DOING BRAIN STUFF NEED TO THINK ABOUT THIS. 
        self.chunked_time = np.zeros(chunked_timestamps.shape)
        for i in range(len(self.patch)):
            self.chunked_time[i,:] = (chunked_timestamps[i,:] - chunked_timestamps[i,0])/60e6
            
class DAQ_Tracking_PURE(DAQ_Trigger):
        def __init__(self,sig,header):
            DAQ_Trigger.__init__(self, sig, header) #Share init conditions from Parent Class (DAQ_Trigger)
            self.chunked_field = self.chunked_data*0.071488e-9
#%% Load in objects in order
    #0nT: up,down; 60nT up,down

for i in updown:

    daq_g = 'Z:/Data/2023_02_22_fake_brain_tests/grad_b_{}_000/'.format(i)
        
    trace_g, trace_legends_g = DAQ_read_shift(daq_g,csv_sep)

    traces_g.append(DAQ_Tracking_PURE(trace_g,trace_legends_g))
    

#%% Add spectra and spectral domain to each object

sec1 = 837

# extract from 3-4 second

ROI_up = np.zeros((7,16))
ROI_down = np.zeros((7,16))

for j in range(0,7):
    sind = sec1*(j+1)
    find = sec1*(j+2)
    for i in traces_g[0].patch:
            ROI_up[j,i] = np.std(traces_g[0].chunked_data[i,int(sind):int(find)])
            ROI_down[j,i] = np.std(traces_g[1].chunked_data[i,int(sind):int(find)])
    

#Average over seconds
ROI_all = np.vstack((ROI_up,ROI_down[:,::-1]))

avg_ROI = np.mean(ROI_all,axis=0)

err = np.std(ROI_all,axis=0)

###########################################

plt.figure()
plt.errorbar(A_Param,avg_ROI,fmt='-o',yerr=err)
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlabel('Applied DC noise, b (nT)')
plt.ylabel('Sensitivity (Shift)/rHz')
plt.xlim(-0.5,5.5)
plt.title('Gradiometer Sensitivity against DC applied noise, b')
plt.show()
#%%


plt.figure()
for i in range(0,14):
    plt.plot(A_Param,ROI_all[i,:],'-o')
   
plt.show()
