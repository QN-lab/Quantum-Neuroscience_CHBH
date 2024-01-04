# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:29:42 2023

@author: kowalcau
"""


import os
import mne
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.get_option("display.max_columns",15)
base_directory='Z:\\Data\\2023_07_28\\Z\\run\\C1_lifts\\'
os.chdir(base_directory)

P='50m'#dir_name[6:]
P1=50 #here in mili
BW='50'#dir_name[0:2]

#####################  ********** ########################

save_directory_A=base_directory[:19]+'\\Analysis' 

scal_fac=round(-299*P1**(-0.779),2)
print('P is '+str(P) +', BW is '+str(BW) +', SF is '+str(scal_fac))

dir_name = os.path.basename(base_directory)

plot_title='AER Grad'+' (SF='+str(scal_fac)+', P='+str(P)+'m, I=100u, BW='+str(BW)+'), ' +'\n'+str(dir_name)

##################### Raw - Our sensor ###########################
#%%
data_g=pd.read_csv('_f.csv',sep=',')
#print(data_g.head(3))
#data_g.shape

data_g['B_T_cal']=data_g['B_T (pT)']*scal_fac*1e-12
data_g['Trig_in2']=data_g['Trig_in2'].round(0).astype(int)
data_g['Aux1_v']=data_g['Aux1_v'].round(0).astype(int)

print(data_g.head(1))
data_g.to_csv("_g_r_raw.csv", sep=',', index=False)

ch_names = ['chunk', 'value', 'time', 'B_T (pT)','error_deg', 'Aux1_v', 'Aux_2', 'Trig_in2','Demod_X', 'Demod_Y', 'Stim','B_T_cal']
ch_types_g = ['misc', 'misc', 'misc','misc', 'misc','eeg','stim', 'misc', 'misc', 'misc', 'misc','mag']

data_raw_g=data_g.T
sfreq_g=1/data_g['time'].iloc[1]
#sfreq_g=837.1
print(sfreq_g)
info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types_g, sfreq=sfreq_g)
raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)
#%%

reject_criteria = dict(mag=20e-5)
reject_criteria_f = dict(mag=20e-5)
l_freq_r = 0.01 ##filter settings for the raw #(doesn't do much right now)
h_freq_r =40
l_freq = 0.01 ##fiter settings for evoked data
h_freq = 30
tmin_e = -0.2
tmax_e = 1
baseline =(-0.2,0)
picks_fl = ['FL0101-BZ_CL']

events_g = mne.find_events(raw_g, stim_channel=['Stim'], min_duration=0.01, mask_type='not_and', \
                                     mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)
#mne.viz.plot_events(events_all_g,first_samp=3000 ) 

raw_gf = raw_g.copy().filter(l_freq=l_freq_r,h_freq=h_freq_r)


epochs_gf=mne.Epochs(raw_gf,picks=['mag'], events=events_g, event_id=3, tmin=tmin_e, tmax=tmax_e,baseline=baseline, reject=reject_criteria, preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)
evoked_gf=epochs_gf.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)

N_events=len(epochs_gf)#or events_gf.shape[0]
print(N_events)
evoked_gf.plot()

##plot  to check if the stim chanell is rejecting trials with the temperatrue sensing on
raw_gf.copy().pick_types(meg=True, stim=True).plot(events=events_g, start=0, duration=20, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'), event_color={ 3:'w'})

#%%
g_epochs_data=np.squeeze(epochs_gf.get_data())
g_epochs_DF=pd.DataFrame(data=(g_epochs_data.T))

#g_epochs_DF.insert(0, 'time', epochs_gf.times)
#g_epochs_DF.to_csv(str(dir_name)+"_origin.csv", sep=',', index=False)
#g_epochs_DF.plot(x='time', legend=None)

g_epochs_DF.iloc[:,:].plot(legend=None, title='Gradiometer' )

#g_epochs_DF.to_csv("epochs_origin.csv", sep=',', index=False)
#g_av=g_epochs_DF.iloc[:,350:].mean(axis=1)

g_av=g_epochs_DF.mean(axis=1)
g_std=g_epochs_DF.std(axis=1)
g_sem=g_epochs_DF.sem(axis=1)
g_epochs_DF.shape                               
g_epochs_DF.head(3)

evoked_gf=epochs_gf.copy().average()
evoked_gf.shapeN_events=len(epochs_gf)#or events_gf.shape[0]
print(epochs_gf)

info_ev=evoked_gf.info
evoked_g2=mne.EvokedArray(np.array([g_av]), info_ev, tmin=-0.2)

g_av=np.squeeze(evoked_g2.filter(l_freq=l_freq,h_freq=h_freq).get_data())


