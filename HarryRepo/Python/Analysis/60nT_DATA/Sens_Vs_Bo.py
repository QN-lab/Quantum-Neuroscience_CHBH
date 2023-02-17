# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:32:25 2023

@author: H
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
import math
import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import statistics as stats

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

class DAQ_Spectrum_PURE(DAQ_Trigger):
    
    def __init__(self,sig,header):

        DAQ_Trigger.__init__(self,sig,header) #Share init conditions from the other child class
 
        pull_rel_off = self.header['grid_col_offset'].tolist()
        self.frq_domain = np.linspace(pull_rel_off[0],-pull_rel_off[0],self.ChunkSize[0])

        sind1 = 0     #-420
        find1 = 25    #-400

        self.floor = np.zeros(len(self.patch))
        self.floor_repd = np.zeros((len(self.patch),self.chunked_data.shape[1]))
        self.max_spect_val = np.zeros(len(self.patch))
        self.SNr = np.zeros(len(self.patch))

        for i in self.patch:
            self.floor[i] = stats.median(self.chunked_data[i,sind1:find1])
            self.floor_repd[i,:] = self.floor[i]*np.ones(8191)
            self.max_spect_val[i] = max(self.chunked_data[i,:])
            self.SNr[i] = self.max_spect_val[i]/self.floor[i]

csv_sep = ';'
A = np.array([0,5,10,30,60])

daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230215_232845_01/grad_lock_BG_000/'
daq_res = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230215_232845_01/grad_lock_BG_res_000/'

trace_0, trace_legends_0 = HCA.DAQ_spect_read(daq,csv_sep) #Read in relevant CSVs as Dataframes
res_0, res_legends_0 = HCA.Res_read(daq_res,csv_sep)

spect_0 = DAQ_Spectrum_PURE(trace_0,trace_legends_0) #create object
resonances = HCA.Resonance(res_0, res_legends_0, 800)

plt.figure()
resonances.plot_with_fit()

g=1/2
hbar=1.05e-34
mu=9.27e-24
sens_0=(2*math.pi*resonances.width*hbar)/(g*mu*spect_0.SNr)
sens_err = (2*math.pi*resonances.width_err*hbar)/(g*mu*spect_0.SNr)
#Sensitivity
plt.figure()
plt.errorbar(A, sens_0/1e-15, yerr=sens_err, fmt='o')
plt.ylabel('Sensitivity (fT/rHz)')
plt.xlabel('Applied Background DC')
# plt.legend()
plt.xlim(-5,65)
# plt.ylim()
plt.grid(color='k', linestyle='-', linewidth=0.2)


