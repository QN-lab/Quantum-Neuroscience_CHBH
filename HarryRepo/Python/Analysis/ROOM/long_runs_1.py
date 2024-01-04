import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('classic')
import math
import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import pandas as pd


csv_sep = ';'

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
            
            

daq_g = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230309_154109_04/40mVpp_aHH_gradient_2_000/'
    
trace_g, trace_legends_g = DAQ_read_shift(daq_g,csv_sep)

traces_g = DAQ_Tracking_PURE(trace_g,trace_legends_g)


fs = 837.1

xf_chunked = np.zeros((len(traces_g.patch),419))
yf_chunked = np.zeros((len(traces_g.patch),419))

for i in traces_g.patch:
    
    xf,yf = signal.welch(traces_g.chunked_data[i,:],fs,nperseg = fs)
    
    xf_chunked[i,:] = xf
    yf_chunked[i,:] = 10*np.log10(yf) #dB!


plt.figure()
for i in traces_g.patch:
    plt.plot(xf,yf_chunked[i,:])
    plt.xlim(0,50)
    
yf_new = np.mean(yf_chunked,axis=0)

plt.figure()
plt.plot(xf,yf_new)
plt.xlim(0,50)
