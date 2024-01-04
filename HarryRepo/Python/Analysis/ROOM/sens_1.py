# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:08:32 2023

@author: H
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
import math
# import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd

A_Param = range(0,50)

csv_sep = ';'

looper_A = range(len(A_Param))


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

daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230309_230503_06/good_grad_empty_1_000/'
    
trace_g, trace_legends_g = DAQ_read_shift(daq_g,csv_sep)

traces_g = DAQ_Tracking_PURE(trace_g,trace_legends_g)

#%% Add spectra and spectral domain to each object

sec1 = 837

Roi = np.zeros((2,50))

for j in range(0,2):
    sind = sec1*(j+1)
    find = sec1*(j+2)
    for i in traces_g.patch:
            Roi[j,i] = np.std(traces_g.chunked_data[i,int(sind):int(find)])

avg_ROI = np.mean(Roi,axis=0)

err = np.std(Roi,axis=0)

###########################################

plt.figure()
plt.errorbar(np.array(A_Param)+1,avg_ROI,fmt='-o',yerr=err)
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlabel('Run number')
plt.ylabel('Sensitivity (Shift)/rHz')
# plt.xlim(-1,16)
plt.title('Sensitivity of gradiometer in a long run')

#%%

# daq_g = 'Z:/jenseno-opm/Data/2023_03_03/Zurich_000/'

# trace_g, trace_legends_g = DAQ_read_shift(daq_g,csv_sep)

# traces_g = DAQ_Tracking_PURE(trace_g,trace_legends_g)
    

#%% Add spectra and spectral domain to each object

# sec1 = 837

# ROI = np.zeros((7,15))

# for j in range(0,7):
#     sind = sec1*(j+1)
#     find = sec1*(j+2)
#     for i in traces_g.patch:
#             ROI[j,i] = np.std(traces_g.chunked_data[i,int(sind):int(find)])
            
    


# avg_ROI = np.mean(ROI,axis=0)

# err = np.std(ROI,axis=0)


# plt.errorbar(np.array(A_Param)+1,avg_ROI,fmt='-o',yerr=err,label = 'Zurich Aux Output')
# plt.legend()
# plt.show()