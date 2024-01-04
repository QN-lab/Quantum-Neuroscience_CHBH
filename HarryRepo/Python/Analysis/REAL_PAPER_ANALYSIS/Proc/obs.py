import matplotlib.pyplot as plt
# plt.style.use('classic')
import numpy as np
import regex as re
import pandas as pd
from scipy import signal
import os
import math
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

###################################################################################################
#%%
#PREPROCESSING

def ReadData(cur_fold,sigs,headers):
    #Read in signals and headers
    out_sig = list()
    out_headers = list()
    for i in range(len(sigs)):
        out_sig.append(pd.read_csv(cur_fold+sigs[i],sep=';'))
        out_headers.append(pd.read_csv(cur_fold+headers[i],sep=';'))
    return out_headers, out_sig
 

def pull_headers(header_df):
    #Extract data from DataFrame into arrays to be mounted to the objects later
    chunk_size = header_df['chunk_size'].tolist()
    chunk_num = header_df['chunk_number'].tolist()
    run_name = header_df['history_name'].tolist()
    return chunk_size, chunk_num, run_name

def pull_signals(sig_df):
    chunk = sig_df['chunk'].tolist()
    timestamps = np.array(sig_df['timestamp'].tolist())
    data = sig_df['value'].tolist()
    
    return chunk, timestamps, data

###################################################################################################
#General Purpose pre-processing
def Janitor(dirty_chunked_data, trigger_chunked_data, patch):
    #Takes dirty data containing time-matched arduino triggers and removes them into a new variable. 
    
    cleaned_data = dirty_chunked_data
    
    for q in patch:
        if (trigger_chunked_data[q,:]>= 4.5).any(): 
            cleaned_data[q,:] = 0 #somehow changes the raw data to have 0 in these spots
            
    cleaned_data = cleaned_data[~np.all(cleaned_data == 0, axis=1)]
    
    return cleaned_data

###################################################################################################
#%%
#FREQUENY-DOMAIN PROCESSING

def Powerise(field_data,chunked_time,chunk): #Power in dB (/rHz?)
    #Converts input data (field or XiY) to power in dB
    
    N = field_data.shape[1]
    T = chunked_time[1]-chunked_time[0]
    xf = fftfreq(N,T)[:N//2]
    
    yf_chunked = np.zeros((len(chunk),len(xf)))

    for a in chunk:
        
        yf = (1/(np.sqrt(837.1/N)))*(np.sqrt(2)/(N))*fft(field_data[a,:])
    
        yf_chunked[a,:] = 20*np.log10(np.abs(yf[0:N//2])) #Field?

    return xf, yf_chunked

def Spectrumise(field_data,chunked_time,chunk):
    #Converts input data (field or XiY) to amplitude spectrum
        #field_data: chunked(trial-by-trial) array of timecourse data
        #chunked_time: chunked time array
        # chunk: list of enumerated trials
    N = field_data.shape[1] #Number of datapoints
    T = chunked_time[1]-chunked_time[0] #Time increment
    xf = fftfreq(N,T)[:N//2] #Frequency domain
    
    yf_chunked = np.zeros((len(chunk),len(xf)))

    for a in chunk:
        
        # yf = ((np.sqrt(2))/(N*np.sqrt(837.1/N)))*fft(field_data[a,:])
        yf = (1/(np.sqrt(837.1/N)))*(np.sqrt(2)/(N))*fft(field_data[a,:])
        yf_chunked[a,:] = np.abs(yf[0:N//2]) #Single-sided spectrum
    
    return xf, yf_chunked

###################################################################################################
#%%
#OBJECT MOUNTING

#General Purpose class
class Data_extract:
        
    def __init__(self, mounted_headers,mounted_sigs): 
        self.chunk_size, self.chunk_num, self.run_name = pull_headers(mounted_headers)
        self.chunk, self.timestamps, self.data = pull_signals(mounted_sigs)
        
        self.chunked_data = np.array(self.data).reshape(len(self.chunk_num),self.chunk_size[0])
        
        chunked_timestamps = self.timestamps.reshape(len(self.chunk_num),self.chunk_size[0])

        self.chunked_time = np.zeros(chunked_timestamps.shape)
        for i in range(len(self.chunk_num)):
            self.chunked_time[i,:] = (chunked_timestamps[i,:] - chunked_timestamps[i,0])/60e6
###################################################################################################
#Main class: field conversion and all spectrum processing

class PiD_processing(Data_extract):
    def __init__(self,mounted_headers,mounted_sigs,Trigger_obj,Gain,T1,T2):
        
        Data_extract.__init__(self, mounted_headers,mounted_sigs)
        
        #Cleaning
        self.clean_chunked_data = Janitor(self.chunked_data,Trigger_obj.chunked_data,self.chunk_num)
        self.Field = self.clean_chunked_data*Gain*0.071488e-9
        self.clean_chunks = list(range(len(self.clean_chunked_data)))
        
        #######################################################################
        #Spectrum processing 
        
        self.Roi_sidx = np.searchsorted(self.chunked_time[0,:], T1, side="left")#remember to correct for 200ms trigger if needed
        # self.Roi_fidx = np.searchsorted(self.chunked_time[0,:], T1+1, side="left")
        self.Roi_fidx = np.searchsorted(self.chunked_time[0,:], T1+1, side="left") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #Change back when not doing DC
        self.Q_sidx = np.searchsorted(self.chunked_time[0,:], T2, side="left")
        self.Q_fidx = np.searchsorted(self.chunked_time[0,:], T2+1, side="left")
        
        self.RoI = self.Field[:,self.Roi_sidx:self.Roi_fidx]
        self.quiet_region = self.Field[:,self.Q_sidx:self.Q_fidx] #CHANGE TO END AFTER 1 SECOND BECAUSE THE END OF TRIALS CONTAIN OSCILLATIONS
        
        self.xf, self.yf_chunked = Powerise(self.RoI,self.chunked_time[0,self.Roi_sidx:self.Roi_fidx],self.clean_chunks)
        
        self.xf_a, self.yf_chunked_a = Spectrumise(self.RoI,self.chunked_time[0,self.Roi_sidx:self.Roi_fidx],self.clean_chunks)
        self.xf_q, self.yf_chunked_q = Spectrumise(self.quiet_region,self.chunked_time[0,self.Q_sidx:self.Q_fidx],self.clean_chunks)
        self.xf_flat_q, self.yf_flat_q = Spectrumise(np.expand_dims(self.quiet_region.flatten(),axis=0),self.chunked_time[0,self.Q_sidx:self.Q_fidx],[0])
        
        allfield = np.expand_dims(self.Field.flatten(),axis=0)
        self.xf_full, self.yf_full = Spectrumise(allfield[:,8371:2*8371],self.chunked_time[0,:],[0]) #Flatten 10s of data for spectrum, this requires a drop.duplicates
        
        if self.yf_chunked.shape[0] >= 10:
            self.yf_avg = np.mean(self.yf_chunked[:9,:],axis=0)
            self.yf_std = np.std(self.yf_chunked[:9,:],axis=0)
            
            self.yf_avg_a = np.mean(self.yf_chunked_a[:9,:],axis=0)
            self.yf_avg_q = np.mean(self.yf_chunked_q[:9,:],axis=0)
        
        else:
            self.yf_avg = np.mean(self.yf_chunked,axis=0)
            self.yf_std = np.std(self.yf_chunked,axis=0)
            
            self.yf_avg_a = np.mean(self.yf_chunked_a,axis=0)
            self.yf_avg_q = np.mean(self.yf_chunked_q,axis=0)
        
        
    def findlmax(self,freq):
        
        idx = (np.abs(self.xf-freq)).argmin()
        maxval = np.zeros(len(self.clean_chunks))
    
        for a in self.clean_chunks:
        
            roi = self.yf_chunked[a,int(idx-2):int(idx+2)]
            maxval[a] = np.max(roi)
        
        return maxval

###################################################################################################

#Copy of above, but for XiY

class XiY_processing(Data_extract):
    def __init__(self,mounted_headers,mounted_sigs,Trigger_obj,T1,T2):
        
        Data_extract.__init__(self, mounted_headers,mounted_sigs)
        
        #Clean Chunks
        self.clean_chunked_data = Janitor(self.chunked_data,Trigger_obj.chunked_data,self.chunk_num)
        self.clean_chunks = list(range(len(self.clean_chunked_data)))
        
        #######################################################################
        #Spectrum processing 
        
        self.Roi_sidx = np.searchsorted(self.chunked_time[0,:], T1, side="left") #correct for 200ms triggerself.Roi_sidx = np.searchsorted(self.chunked_time[0,:], T1, side="left") #correct for 200ms trigger
        self.Roi_fidx = np.searchsorted(self.chunked_time[0,:], T2, side="left")
        
        self.RoI = self.clean_chunked_data[:,self.Roi_sidx:self.Roi_fidx]
        self.quiet_region = self.clean_chunked_data[:,self.Roi_fidx+1:] #CHANGE TO END AFTER 1 SECOND BECAUSE THE END OF TRIALS CONTAIN OSCILLATIONS
        
        self.xf, self.yf_chunked = Spectrumise(self.RoI,self.chunked_time[0,self.Roi_sidx:self.Roi_fidx],self.clean_chunks)
        
        if self.yf_chunked.shape[0] >= 10:
            self.yf_avg = np.mean(self.yf_chunked[:9,:],axis=0)
            self.yf_std = np.std(self.yf_chunked[:9,:],axis=0)
        
        else:
            self.yf_avg = np.mean(self.yf_chunked,axis=0)
            self.yf_std = np.std(self.yf_chunked,axis=0)
        
        self.floor_f1 = 75
        self.floor_f2 = 95
        self.floor_sind = np.searchsorted(self.xf, self.floor_f1, side="left")
        self.floor_find = np.searchsorted(self.xf, self.floor_f2, side="left")
        
        self.SnR = np.max(self.yf_avg)/np.mean(self.yf_avg[self.floor_f1:self.floor_f2])
    
    def findlmax(self,freq):
        
        idx = (np.abs(self.xf-freq)).argmin()
        maxval = np.zeros(len(self.clean_chunks))
    
        for a in self.clean_chunks:
        
            roi = self.yf_chunked[a,int(idx-2):int(idx+2)]
            maxval[a] = np.max(roi)
        
        return maxval
        
###############################################################################
#%% 
#Combine all data objects into a single object.
class Joined:
    
    def __init__(self,base_directory,subfolder,Gain,T1,T2):
        
        #Gain: PLL Gain Value
        #T1: Time in seconds within each trial to begin the 'ON' Period, 
            #i.e. the time in which we are applying something to the sensor
            
        #T2: Time to be begin the 'OFF' period, where there is no applied
        
        headers = ['dev3994_demods_0_sample_auxin1_avg_header_00000.csv',
                 'dev3994_demods_0_sample_trigin2_avg_header_00000.csv',
                 'dev3994_demods_0_sample_x_avg_header_00000.csv',
                 'dev3994_demods_0_sample_y_avg_header_00000.csv',
                 'dev3994_pids_0_stream_shift_avg_header_00000.csv'
                 ]

        sigs = ['dev3994_demods_0_sample_auxin1_avg_00000.csv',     #Arduino
              'dev3994_demods_0_sample_trigin2_avg_00000.csv',      #Trigger
              'dev3994_demods_0_sample_x_avg_00000.csv',            #X
              'dev3994_demods_0_sample_y_avg_00000.csv',            #Y
              'dev3994_pids_0_stream_shift_avg_00000.csv'           #PID Signal
                  ]
        
        
        cur_fold = base_directory+subfolder+'\\'
        mounted_headers, mounted_sigs = ReadData(cur_fold,sigs,headers)
        
        self.subfolder_name = subfolder
        self.Arduino = Data_extract(mounted_headers[0],mounted_sigs[0])
        self.Trigger_in = Data_extract(mounted_headers[1],mounted_sigs[1])
        self.X = XiY_processing(mounted_headers[2],mounted_sigs[2],self.Arduino,T1,T2)
        self.Y = XiY_processing(mounted_headers[3],mounted_sigs[3],self.Arduino,T1,T2)
        self.PiD = PiD_processing(mounted_headers[4],mounted_sigs[4],self.Arduino,Gain,T1,T2)
        
    #Plotting.
    
    def plotpower(self,fig,ax):
        #Plots all spectra, no average spectra
        ax.stackplot(self.PiD.xf,self.PiD.yf_chunked)
        # ax.set_yscale('log')
        ax.set_xlim([0, 100])
        ax.set_title(self.subfolder_name)
    
    def plotavgpower(self,fig,ax):
        #Plot average spectrum, and titles with filename
        ax.plot(self.PiD.xf,self.PiD.yf_avg)
        # ax.set_yscale('log')
        ax.set_xlim([0, 100])
        ax.set_title(self.subfolder_name)
        
    def plotavgpower_title(self,fig,ax,title):
        #Same as above, with custom title, Needed?
        ax.plot(self.PiD.xf,self.PiD.yf_avg)
        # # ax.set_yscale('log')
        ax.set_xlim([0, 100])
        ax.set_title(title)
        
    def plot_xy_power(self,fig,ax,c):
        #Plot X on + and Y on - (THIS IS NOT X+iY SPECTRUM)
        ax.plot(self.X.xf,self.X.yf_avg,c)
        ax.plot(self.Y.xf*-1,self.Y.yf_avg,c)
        ax.set_yscale('log')
        # ax.set_xlim([0, 100])
        ax.set_title(self.subfolder_name)
        
    def Noise_spectrum_title(self,title,xscale):
        #Plot noise spectrum in T (T/rHz if over 1s)
        fig,axs = plt.subplots(2,1,constrained_layout=True)
        
        fig.suptitle('Noise Spectra: '+ title,fontsize=15)
        
        axs[0].plot(self.PiD.xf_a,self.PiD.yf_avg_a,linewidth=2)
        
        axs[0].grid(True)
        axs[0].set_title('Spectrum during ON')
        axs[0].set_ylabel('Field(T)')
        axs[0].set_yscale('log')
        axs[0].set_xscale(xscale)
        # axs[0].set_xlim((10**-1,10**3))
        # axs[0].set_ylim((10**-15,10**-10))
    
        
        axs[1].plot(self.PiD.xf_q,self.PiD.yf_avg_q,linewidth=2)
        
        axs[1].grid(True)
        
        axs[1].set_title('Spectrum during OFF')
        axs[1].set_ylabel('Field(T)')
        axs[1].set_xlabel('Frequency(Hz)')
        axs[1].set_yscale('log')
        axs[1].set_xscale(xscale)
        # axs[1].set_ylim((10**-15,10**-10))
        # axs[1].set_xlim((10**-1,10**3))
    
        
#%% Sensitivity

# def sens_SNR(Joined,Resonance):
#     g=1/2
#     hbar=1.05e-34
#     mu=9.27e-24
    
#     sens = (2*math.pi*hbar*np.mean(Resonance.width))/(g*mu*Joined.X.SnR)
#     return sens

def sens_std(Joined):
    # std dev over 1 second
    
    r_sens_i = np.std(Joined.PiD.RoI,axis=1)
    avg_r_sens = np.mean(r_sens_i)
    std_r_sens = np.std(r_sens_i)
    
    q_sens_i = np.std(Joined.PiD.quiet_region,axis=1)
    avg_q_sens = np.mean(q_sens_i)
    std_q_sens = np.std(q_sens_i)
    
    r_sens = [avg_r_sens, std_r_sens]
    q_sens = [avg_q_sens,std_q_sens]
    
    return r_sens,q_sens
    
def all_sens_plot(fig,ax,xdat,ydat,xscale,title):
    
    fig.suptitle('Noise Spectrum: '+ title,fontsize=15)
    
    ax.plot(xdat,ydat,linewidth=2)
    
    ax.grid(True)
    ax.set_ylabel('Field(T)')
    # ax.
    ax.set_yscale('log')
    ax.set_xscale(xscale)
#%% Send data to CSV (change to desired directory first)

def ToCSV(array,headers,filename):
    df = pd.DataFrame(array,columns=headers)
    df.to_csv(filename)


#%% Fitting and plotting Resonance files

def resReadData(folder_path,filename,headername,csv_sep):
    out_sig = pd.read_csv(folder_path+filename,sep=csv_sep)
    out_headers = pd.read_csv(folder_path+headername,sep=csv_sep)
    return out_sig, out_headers

def Lorentzian(x, amp, cen, wid, slope, offset):
    return (amp*(wid)**2/((x-cen)**2+(wid)**2)) + slope*x + offset

class Resonance:
    def __init__(self,sig,header,sfreq):
      self.sig=sig
      self.header=header
      
      #Pull from headers
      self.ChunkSize = self.header['chunk_size'].tolist() #Prints the size of each chunk
      self.patch = self.header['chunk_number'].tolist() # Prints a list of the number of runs
      self.run_names = self.header['history_name'].tolist()
      
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
      self.peakindx = start_indx + self.data[:,start_indx:,1].argmax(axis=1) #index at which the lorentzian peak exists for each run
      
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
          popt_lor, pcov_lor = curve_fit(Lorentzian,data_red[:,0],data_red[:,1]*1000,p0=[1,self.data[j,self.peakindx[j],0],200,0,0]) #guess peak: min-max
          self.fit_cov_mat[j,:,:] = pcov_lor
          self.fit_params[j,:]=popt_lor
          self.field_res[j]=71*1e-6*self.fit_params[j,1]
          
          
      #Identifying Params
      self.amplitude = self.fit_params[:,0]
      self.central_f = self.fit_params[:,1]
      self.width=2*abs(self.fit_params[:,2]) #Twice half width
          
      #Acquire Fit Errors for each run
      self.fiterr = np.zeros((len(self.patch),5))

      for i in range(len(self.patch)):
          self.fiterr[i,:] = np.sqrt(np.diag(self.fit_cov_mat[i,:,:]))
          
      #Identifying Params
      self.amplitude_err = self.fiterr[:,0]
      self.central_f_err = self.fiterr[:,1]
      self.width_err=2*self.fiterr[:,2] #Twice half width
      
      #Height/width
      self.h_over_w = self.amplitude/self.width
      self.h_over_w_err = self.h_over_w*np.sqrt(((self.width_err/self.width)**2)+((self.amplitude_err/self.amplitude)**2))
          
    def plot_with_fit(self):
          #Requires plot to be created
          for i in self.patch:
              #Plot Target data
              plt.plot(self.data[i,:,0], self.data[i,:,1]*1000,color="black")
              plt.ylabel("quadrature, mV")
                  
              #Plot the Fit
              plt.plot(self.data[i,:,0],Lorentzian(self.data[i,:,0], *self.fit_params[i,:]),label=str(self.run_names[i]))
              plt.xlabel("frequency, Hz")
              plt.grid(color='k', linestyle='-', linewidth=0.5)
              # plt.legend(loc='upper right')
              
    def plot_no_fit(self):
        for i in self.patch:
            #Plot Target data
            plt.plot(self.data[i,:,0], self.data[i,:,1]*1000,label=str(self.run_names[i]))
            plt.ylabel("quadrature, mV")
            plt.grid(color='k', linestyle='-', linewidth=0.5)
            # plt.legend(loc='upper left')
            
    def plot_both(self):
        plt.figure()
        ax=plt.gca()
        for i in self.patch:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(self.data[i,:,0], self.data[i,:,1]*1000,label=str(self.run_names[i]),color=color)
            plt.plot(self.data[i,:,0], self.data[i,:,2]*1000,color=color)
            plt.ylabel("mV")
            plt.xlabel("frequency (Hz)")
            plt.grid(color='k', linestyle='-', linewidth=0.5)
            plt.legend(loc='upper right')
            
    def plot_both_1run(self,fig,ax):
        
        ax.plot(self.data[0,:,0], self.data[0,:,1]*1000,color='b')
        ax.plot(self.data[0,:,0], self.data[0,:,2]*1000,color='r')
        ax.set_ylabel("Signal (mV)")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(color='k', linestyle='-', linewidth=0.5)
        # plt.legend(loc='upper right')

    def plot_Y(self):
        plt.figure()
        ax=plt.gca()
        for i in self.patch:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(self.data[i,:,0], self.data[i,:,2]*1000,label=str(self.run_names[i]),color=color)
            plt.ylabel("mV")
            plt.xlabel("Frequency (Hz)")
            plt.grid(color='k', linestyle='-', linewidth=0.5)
            plt.legend(loc='upper right')



#%%

def PhaseReadData(folder_path,filename,headername,csv_sep):
    out_sig = pd.read_csv(folder_path+filename,sep=csv_sep)
    out_headers = pd.read_csv(folder_path+headername,sep=csv_sep)
    return out_sig, out_headers

def Sigmoid(x, alpha, beta):
    return ((2*beta)/(1+math.exp(-alpha*x)))-1*beta

class Phase:
    def __init__(self,sig,header,sfreq):
      self.sig=sig
      self.header=header
      
      #Pull from headers
      self.ChunkSize = self.header['chunk_size'].tolist() #Prints the size of each chunk
      self.patch = self.header['chunk_number'].tolist() # Prints a list of the number of runs
      self.run_names = self.header['history_name'].tolist()
      
      x_ind = self.sig.index[self.sig['fieldname'] == 'phase'].tolist()
      x_data = self.sig.iloc[x_ind,4:-1]
      
      #Save as single Data Array
      self.data=np.zeros((len(self.patch),frq_data.shape[1],4))
      for x in self.patch:
          self.data[x,:,0]=np.array(frq_data.iloc[x,:])
          self.data[x,:,1]=np.array(x_data.iloc[x,:])
          self.data[x,:,2]=np.array(y_data.iloc[x,:]) 
          

    def plot_with_fit(self):
          #Requires plot to be created
          for i in self.patch:
              #Plot Target data
              plt.plot(self.data[i,:,0], self.data[i,:,1]*1000,color="black")
              plt.ylabel("quadrature, mV")
                  
              #Plot the Fit
              plt.plot(self.data[i,:,0],Lorentzian(self.data[i,:,0], *self.fit_params[i,:]),label=str(self.run_names[i]))
              plt.xlabel("frequency, Hz")
              plt.grid(color='k', linestyle='-', linewidth=0.5)
              # plt.legend(loc='upper right')
              
    def plot_no_fit(self):
        for i in self.patch:
            #Plot Target data
            plt.plot(self.data[i,:,0], self.data[i,:,1]*1000,label=str(self.run_names[i]))
            plt.ylabel("quadrature, mV")
            plt.grid(color='k', linestyle='-', linewidth=0.5)
            # plt.legend(loc='upper left')
            
    def plot_both(self):
        plt.figure()
        ax=plt.gca()
        for i in self.patch:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(self.data[i,:,0], self.data[i,:,1]*1000,label=str(self.run_names[i]),color=color)
            plt.plot(self.data[i,:,0], self.data[i,:,2]*1000,color=color)
            plt.ylabel("mV")
            plt.xlabel("frequency (Hz)")
            plt.grid(color='k', linestyle='-', linewidth=0.5)
            plt.legend(loc='upper right')
            
    def plot_both_1run(self,fig,ax):
        
        ax.plot(self.data[0,:,0], self.data[0,:,1]*1000,color='b')
        ax.plot(self.data[0,:,0], self.data[0,:,2]*1000,color='r')
        ax.set_ylabel("Signal (mV)")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(color='k', linestyle='-', linewidth=0.5)
        # plt.legend(loc='upper right')

    def plot_Y(self):
        plt.figure()
        ax=plt.gca()
        for i in self.patch:
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(self.data[i,:,0], self.data[i,:,2]*1000,label=str(self.run_names[i]),color=color)
            plt.ylabel("mV")
            plt.xlabel("Frequency (Hz)")
            plt.grid(color='k', linestyle='-', linewidth=0.5)
            plt.legend(loc='upper right')










    
