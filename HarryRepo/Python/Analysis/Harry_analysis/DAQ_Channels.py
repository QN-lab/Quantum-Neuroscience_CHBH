import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import statistics as stats
import pandas as pd

#Janitor (removes runs that contain the trigger caused by temperature sampling)
def clean_from_trigger(dirty_chunked_data, trigger_chunked_data, patch):
    cleaned_data = dirty_chunked_data
    
    for q in patch:
        if trigger_chunked_data[q,:].any() == 1:
            cleaned_data[q,:] = 0
            
    cleaned_data = cleaned_data[~np.all(cleaned_data == 0, axis=1)]
    
    return cleaned_data


##CLASSES

#PARENT CLASS, ONLY READS IN AND CHUNKS THE DATA
class DAQ_Trigger:
    def __init__(self,sig,header):
        
        self.sig=sig 
        self.header=header
        
        #Pull from headers
        self.ChunkSize = self.header['chunk_size'].tolist() #Size of each chunk
        self.patch = self.header['chunk_number'].tolist() # Prints a list of the number of runs
        self.run_names = self.header['history_name'].tolist()
        
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

#CHILD 1: SAME AS ABOVE BUT READS IN TRIGGER TO CLEAN ITSELF
class DAQ_Tracking(DAQ_Trigger):
        def __init__(self,sig,header,trigger_object):
            DAQ_Trigger.__init__(self, sig, header) #Share init conditions from Parent Class (DAQ_Trigger)
            
            self.cleaned_chunked = clean_from_trigger(self.chunked_data,trigger_object.chunked_data, trigger_object.patch) #Use cleaner function to remove spoiled data
            
            self.cleaned_chunked_field = self.cleaned_chunked/14.28
            self.cleaned_continuous_field = (self.cleaned_chunked.reshape(-1,1))/14.28 # 14.28 Hz/nT mod freq to field (2x larmor gyromagnetic ratio)
            
class DAQ_Tracking_PURE(DAQ_Trigger):
        def __init__(self,sig,header):
            DAQ_Trigger.__init__(self, sig, header) #Share init conditions from Parent Class (DAQ_Trigger)
            
            

#CHILD 2: SAME AS CHILD 1 BUT ADDS SOME FREQUENCY STUFF.
class DAQ_Spectrum(DAQ_Tracking):
    def __init__(self,sig,header,trigger_object):
        
        DAQ_Tracking.__init__(self,sig,header,trigger_object) #Share init conditions from the other child class
        
        #delete irreleveant fields from inherited class (tracking)
        del self.cleaned_chunked_field
        del self.cleaned_continuous_field
        
        # #Load in filter
        # filtsheet = pd.read_csv('Z:/Github/HarryRepo/Python/Analysis/Harry_analysis/Filter1023.csv',sep=';')
        # self.filter = filtsheet['value'].tolist() #FilterData from sheet, if sampling changes, this must too.
        
        
        #Spectrum frequency domain, pulled from the first run.

        # self.frq_domain = np.linspace(0,self.ChunkSize[0],num=self.ChunkSize[0])
        
        pull_rel_off = self.header['grid_col_offset'].tolist()
        self.frq_domain = np.linspace(pull_rel_off[0],-pull_rel_off[0],self.ChunkSize[0])
        
        sind1 = 0     #-420
        find1 = 25    #-400
        
        self.single_spect = self.cleaned_chunked[0,:]  #/7.14 #CONVERT TO FIELD (CHECK ALWAYS)  #/self.filter # Pulls first clean spectrum from recording. 
        self.single_floor = stats.median(self.single_spect[sind1:find1])
        self.single_floor_repd = self.single_floor*np.ones(self.single_spect.shape)
        self.single_max_spect_val = max(self.single_spect)
        self.single_SNr = self.single_max_spect_val/self.single_floor
        
        #Averaged spectra
        self.avg_spect = np.mean(self.cleaned_chunked,axis=0) #*0.071e-9 # 0.071nT/Hz only if working with PID shift
        
        sind2 = 0         
        find2 = 25  
        
        
        self.avg_floor = stats.median(self.avg_spect[sind2:find2])
            
        self.avg_floor_repd = self.avg_floor*np.ones(self.avg_spect.shape)
        
        self.avg_max_spect_val = max(self.avg_spect)
        self.avg_SNr = self.avg_max_spect_val/self.avg_floor
        
        
        