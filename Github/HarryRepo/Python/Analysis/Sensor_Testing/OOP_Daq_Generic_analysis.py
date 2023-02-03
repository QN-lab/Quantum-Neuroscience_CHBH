# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:32:38 2023

Daq Analysis

@author: H
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import pandas as pd
import math
from scipy.optimize import curve_fit
import statistics as stats 


#%% preamble & User inputs
csv_sep = ';' #separator for saved CSV
sfreq = 2000 #frequency at which to start the fitting

#%% Data Read-In

def ReadData(folder_path,filename,headername):
    output_sig = pd.read_csv(folder_path+filename,sep=csv_sep)
    output_headers = pd.read_csv(folder_path+headername,sep=csv_sep)
    
    return output_sig, output_headers


#Resonance Data
Folderpath_r = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230110_232908_05/C1_res_1_000/'     
Filename_r = 'dev3994_demods_0_sample_00000.csv'
Headername_r = 'dev3994_demods_0_sample_header_00000.csv'

res, res_legends = ReadData(Folderpath_r,Filename_r,Headername_r)

#DAQ Data
Folderpath_daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230110_232908_05/C1_daq_1_000/'

#DAQ Spectrum
Filename_spectrum = 'dev3994_demods_0_sample_xiy_fft_abs_avg_00000.csv' #Spectrum Data
Headername_s = 'dev3994_demods_0_sample_xiy_fft_abs_avg_header_00000.csv' #Headers

spect, spect_legends = ReadData(Folderpath_daq,Filename_spectrum,Headername_s)

#Signal channel
Filename_signal = 'dev3994_demods_0_sample_frequency_avg_00000.csv'
Headername_signal = 'dev3994_demods_0_sample_frequency_avg_header_00000.csv' 

track, track_legends = ReadData(Folderpath_daq,Filename_signal,Headername_signal)

#Trigger
Filename_trigger = 'dev3994_demods_0_sample_trigin2_avg_00000.csv'
Headername_trigger = 'dev3994_demods_0_sample_trigin2_avg_header_00000.csv'

trig, trig_legends = ReadData(Folderpath_daq,Filename_trigger,Headername_trigger)

#%%

def Lorentzian(self,x, amp, cen, wid, slope, offset):
    return (amp*(wid)**2/((x-cen)**2+(wid)**2)) + slope*x + offset

class Resonance:
    def __init__(self,sig,header):
      self.sig=sig
      self.header=header
      
      #Pull from headers
      self.ChunkSize = self.header['chunk_size'].tolist() #Prints the size of each chunk
      self.patch = self.header['chunk_number'].tolist() # Prints a list of the number of runs
      
      x_ind = self.sig.index[self.sig['fieldname'] == 'x'].tolist()
      x_data = self.sig.iloc[x_ind,4:-1]
      
      y_ind = self.sig.index[self.sig['fieldname'] == 'y'].tolist()
      y_data =self.sig.iloc[y_ind,4:-1]
      
      frq_ind = self.sig.index[self.sig['fieldname'] == 'frequency'].tolist()
      frq_data = self.sig.iloc[frq_ind,4:-1]
      
      #Save as single Data Array
      self.data=np.zeros((len(self.patch),frq_data.shape[1],4))
      for x in self.patch:
          self.data[x,:,0]=np.array(frq_data.iloc[x,:])
          self.data[x,:,1]=np.array(x_data.iloc[x,:])
          self.data[x,:,2]=np.array(y_data.iloc[x,:])

     #Automatically create central frequency guess after a certain frequency(ignore zero field)
      start_indx = np.abs(self.data[0,:,0] - sfreq).argmin()
      peakindx = start_indx + self.data[:,start_indx:,1].argmax(axis=1) #index at which the lorentzian peak exists for each run
      
     #Fitting
      k=0;
      self.fit_params = np.zeros((len(self.patch),5)) #(amplitude, central frequency, half width, slope, constant offset)
      self.fit_cov_mat= np.zeros((len(self.patch),5,5))
      self.field_res=np.zeros((len(self.patch),5))
      for j in range(len(self.patch)):
          data_red=[]
          k=0
          for i in range(0,self.data[0,:,1].size-10,10):
              if abs(np.mean(self.data[self.patch[j],i+5:i+10,1])-np.mean(self.data[self.patch[j],i:i+5,1]))>5e-7:
                  if k==0: data_red=np.asarray(self.data[self.patch[j],i:i+10,:]) 
                  else: data_red=np.append(data_red,self.data[self.patch[j],i:i+10,:],axis=0)
                  k+=1
          popt_lor, pcov_lor = curve_fit(Lorentzian,data_red[:,0],data_red[:,1]*1000,p0=[1,self.data[j,peakindx[j],0],200,0,0]) #guess peak: min-max
          self.fit_cov_mat[j,:,:] = pcov_lor
          self.fit_params[j,:]=popt_lor
          self.field_res[j]=71*1e-6*self.fit_params[j,1]
          
          
      #Acquire Fit Errors for each run
      self.fiterr = np.zeros((len(self.patch),5))

      for i in range(len(self.patch)):
          self.fiterr[i,:] = np.sqrt(np.diag(self.fit_cov_mat[i,:,:]))
      
    def plot_with_fits(self):
          #Requires plot to be created
          plt.ion()
          for i in self.patch:
              #Plot Target data
              plt.plot(self.data[i,:,0], self.data[i,:,1]*1000,label='Signal')
              plt.ylabel("quadrature, mV")
                  
              #Plot the Fit
              plt.plot(self.data[i,:,0],Lorentzian(self.data[i,:,0], *self.fit_params[i,:]),label='Fit')
              plt.xlabel("frequency, Hz")
              plt.grid(color='k', linestyle='-', linewidth=0.5)
              plt.legend()
          
          
class DAQ_Recording:
    def __init__(self,sig,header):
        
        self.sig=sig
        self.header=header
        
        #Pull from headers
        self.ChunkSize = self.header['chunk_size'].tolist() #Size of each chunk
        self.patch = self.header['chunk_number'].tolist() # Prints a list of the number of runs
        
        #Pull from Signals
        self.Chunk = self.sig['chunk'].tolist()
        
        self.Data = self.sig['value'].tolist()
        
        #Organise by Run
        self.chunked_data = np.array(self.Data).reshape(len(self.patch),self.ChunkSize[0])

#Call Spectrum as a child class with in-built analysis
class DAQ_Spectrum(DAQ_Recording):
    def __init__(self,sig,header):
        DAQ_Recording.__init__(self,sig,header)
        self.floor = np.zeros(self.chunked_data.shape)
        for i in self.patch:
            self.floor[i,:] = stats.median(self.chunked_data[i,:]) * np.ones(self.ChunkSize[0])
            
        #Spectrum offsets, pulled from the first run, so if runs are significantly different, this needs to be changed.
        pull_rel_off = self.header['grid_col_offset'].tolist()
        self.frq_domain = np.linspace(pull_rel_off[0],-pull_rel_off[0],self.ChunkSize[0])
        
        
        
    
    
    
    
    # def plot_spectra(self,sfreq)
            

#%%ASSIGN TO CLASSES
resonance = Resonance(res, res_legends)

tracking = DAQ_Recording(track,track_legends)
trigger = DAQ_Recording(trig,trig_legends)

spectrum = DAQ_Spectrum(spect,spect_legends)


resonance.plot_with_fits()


#%%Plotting and Rejection of bad runs
tracking.cleaned = tracking.chunked_data
spectrum.cleaned = spectrum.chunked_data

for q in tracking.patch:
    if trigger.chunked_data[q,:].any() == 1:
        spectrum.cleaned[q,:] = 0
        tracking.cleaned[q,:] = 0
        
tracking.cleaned = tracking.cleaned[~np.all(tracking.cleaned == 0, axis=1)]
spectrum.cleaned = spectrum.cleaned[~np.all(spectrum.cleaned == 0, axis=1)]

print('Remaining Runs: {} of {}'.format(tracking.cleaned.shape[0],tracking.chunked_data.shape[0]))

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

xlabel = 'Cell 2 Temp (Cell 1 ~ Cell2 -4C)'

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
plt.xlabel(xlabel)

#Width/height Ratio
slope = param_res[:,0]/width
    #Add errors in quadrature
slope_er= slope*np.sqrt(((wid_er/width)**2)+((amp_er/param_res[:,0])**2))

Fig4 = plt.figure('fig4')
#plt.plot(Pow,param_res[:,0]/width,'bD')
plt.errorbar(Pow,slope,yerr=slope_er,fmt='.b')
plt.ylabel('Amplitude over width (mV/Hz)')
plt.xlabel(xlabel)
plt.grid(color='k', linestyle='-', linewidth=0.5)

#Sensitivity
    #Error
sens_er = (2*math.pi*wid_er*hbar)/(g*mu*SNr) # assumes no error on SNr, probably need to account for this
Fig5 = plt.figure('fig5')
plt.errorbar(Pow,sensitivity,yerr=sens_er,fmt='.b')
plt.ylabel('Sensitivity (T/rHz)')
plt.xlabel(xlabel)
plt.grid(color='k', linestyle='-', linewidth=0.5)

