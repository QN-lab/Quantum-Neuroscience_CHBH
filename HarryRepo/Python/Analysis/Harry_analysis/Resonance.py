from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import math
plt.style.use('classic')



#Fitting function
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
              plt.legend(loc='upper right')
              
    def plot_no_fit(self):
        for i in self.patch:
            #Plot Target data
            plt.plot(self.data[i,:,0], self.data[i,:,1]*1000,label=str(self.run_names[i]))
            plt.ylabel("quadrature, mV")
            plt.grid(color='k', linestyle='-', linewidth=0.5)
            plt.legend(loc='upper left')
                
            
            
class Spect_analyser:
    def __init__(self,sig,header):
        
        self.sig=sig 
        self.header=header
        
        #Pull from headers
        self.ChunkSize = self.header['chunk_size'].tolist() #Size of each chunk
        self.patch = self.header['chunk_number'].tolist() # Prints a list of the number of runs
        self.run_names = self.header['history_name'].tolist()
        
        #Pull from Signals
        self.Chunk = self.sig['chunk'].tolist()
        self.Data = 10*np.log10(np.array(self.sig['value'].tolist())) #CHANGING TO dB
        
        #Organise by Run
        self.chunked_data = self.Data.reshape(len(self.patch),self.ChunkSize[0])
        # self.chunked_data = self.chunked_data[1:,:]  #REMOVES CLEAN RUN, ONLY USE THIS IF DATA CONTAINS IT.
        
        
        pull_rel_off = self.header['grid_col_offset'].tolist()
        self.frq_domain = np.linspace(pull_rel_off[0],-pull_rel_off[0],self.ChunkSize[0])
        
        
    def extract_peaks(self):
        freq_list_temp = []
        self.freq_list = np.zeros(len(self.run_names))
    
        for i in range(len(self.run_names)):
            freq_list_temp = re.findall(r'\d+', self.run_names[i])
            if not freq_list_temp:
                freq_list_temp = [np.nan]
            self.freq_list[i] = freq_list_temp[0]
    
    
        nidx = np.zeros(len(self.freq_list))
        self.maxval = np.zeros(len(self.freq_list))
    
    
        for j in range(len(self.freq_list)):
            if math.isnan(self.freq_list[j]):
                pass
            else:
                nidx[j] = (np.abs(self.frq_domain-self.freq_list[j])).argmin()
                
                roi = self.chunked_data[j,int(nidx[j]-2):int(nidx[j]+2)]
                
                self.maxval[j] = np.max(roi)
                
        return self.maxval, self.freq_list
            
            

        