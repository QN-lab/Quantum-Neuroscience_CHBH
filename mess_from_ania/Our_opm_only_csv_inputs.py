# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:05:28 2023

@author: kowalcau
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:03:48 2023

@author: kowalcau
"""


import os
import mne
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.get_option("display.max_columns",15)
##################### CHANGE HERE IN ALL LINES ###########################################
base_directory='Z:\\Data\\2023_07_03\\Z\\run\\500_50BW_4_L' # MAKE SURE THERE IS NO \\ AT THE END

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
data_g=pd.read_csv('_g_r.csv',sep=',')
#print(data_g.head(3))
#data_g.shape

data_g['B_T_cal']=data_g['B_T (pT)']*scal_fac*1e-12
data_g['Trig_in2']=data_g['Trig_in2'].round(0).astype(int)
data_g['Aux1_v']=data_g['Aux1_v'].round(0).astype(int)

print(data_g.head(1))
data_g.to_csv("_g_r_raw.csv", sep=',', index=False)

ch_names = ['chunk', 'value', 'time', 'B_T (pT)','error_deg', 'Aux1_v', 'Aux_2', 'Trig_in2','Demod_X', 'Demod_Y', 'Stim','B_T_cal']
ch_types_g = ['misc', 'misc', 'misc','misc', 'misc','eeg','stim', 'stim', 'misc', 'misc', 'stim','mag']

data_raw_g=data_g.T
sfreq_g=1/data_g['time'].iloc[1]
#sfreq_g=837.1
print(sfreq_g)
info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types_g, sfreq=sfreq_g)
raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)

######################### Raw - fieldline #############################################

data_fl=pd.read_csv('_fl_r.csv',sep=',')
data_fl['time']=data_fl['time']
data_fl['Grad1']=data_fl['FL0101-BZ_CL']-data_fl['FL0102-BZ_CL']
data_fl['Grad2']=data_fl['FL0103-BZ_CL']-data_fl['FL0104-BZ_CL']
print(data_fl.head(3))
sfreq_fl=1/(data_fl['time'].iloc[1])

ch_names_fl = ['time',   'FL0101-BZ_CL','FL0102-BZ_CL','FL0103-BZ_CL','FL0104-BZ_CL','Input1', 'Stim', 'Grad1', 'Grad2']
ch_types_fl = ['misc',  'mag', 'mag', 'mag','mag','stim', 'stim', 'mag','mag']

data_raw_fl=data_fl.T

info_fl = mne.create_info(ch_names=ch_names_fl, ch_types=ch_types_fl, sfreq=sfreq_fl)
raw_fl= mne.io.RawArray(data_raw_fl, info_fl, verbose=True)


#%% ############### events epoching and rejection

reject_criteria = dict(mag=3000e-12)
reject_criteria_f = dict(mag=3000e-12)
l_freq_r = 0.01 ##filter settings for the raw #(doesn't do much right now)
h_freq_r =100
l_freq = 0.01 ##fiter settings for evoked data
h_freq = 100
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
#evoked_gf.plot()

#plot  to check if the stim chanell is rejecting trials with the temperatrue sensing on
#raw_g.copy().pick_types(meg=True, stim=True).plot(events=events_g, start=0, duration=20, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'), event_color={ 3:'b'})
 

# events_all_fl = mne.find_events(raw_fl, stim_channel=['Input1'], min_duration=0.01,  mask_type='not_and', \
#                                 mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)
# # matched events to our sensor
events_m_fl = mne.find_events(raw_fl, stim_channel=['Stim'], min_duration=0.01,   mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)

#raw_fl.copy().pick_types(meg=True, stim=True).plot(events=events_m_fl, start=0, duration=25, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'), event_color={ 3:'b'})
#mne.viz.plot_events(events_m_fl) 


raw_flf = raw_fl.copy().filter(l_freq=l_freq_r, h_freq=h_freq_r)

epochs_flf=mne.Epochs(raw_flf,events=events_m_fl, picks=picks_fl, tmin=tmin_e,tmax=tmax_e, baseline =baseline, reject=reject_criteria_f,preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)
evoked_flf=epochs_flf.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)
N_events_fl = len(epochs_flf)#events_m_fl.shape[0]

np.shape(evoked_flf)
#
#evoked_flf.plot()

# #evoked_fl_exp = evoked_flf.copy().pick_types(meg=True,stim=False)

#%%
###### plotting
g_epochs_data=np.squeeze(epochs_gf.get_data())
g_epochs_DF=pd.DataFrame(data=(g_epochs_data.T))


# g_epochs_DF.plot(x='time', legend=None)
#g_epochs_DF.to_csv("origin_g.csv", sep=',', index=False)
g_epochs_DF.iloc[:,:].plot(legend=None, title='Gradiometer' )

os.chdir(base_directory)
g_epochs_DF.insert(0, 'time', epochs_gf.times)
g_epochs_DF.to_csv(str(dir_name)+"_epochs_origin_G.csv", sep=',', index=False)
os.chdir(save_directory_A)

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





fl_epochs_data=np.squeeze(epochs_flf.get_data())
fl_epochs_DF=pd.DataFrame(data=(fl_epochs_data[0:].T))
# g_epochs_DF.insert(0, 'time', epochs_gf.times)
fl_av=fl_epochs_DF.mean(axis=1)
fl_std=fl_epochs_DF.std(axis=1)
fl_sem=fl_epochs_DF.sem(axis=1)
# raw_fl.copy().pick_types(meg=True, stim=True).plot(events=events_fl, start=50, duration=50, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'))

fl_epochs_DF.iloc[:,:].plot(legend=None)

info_ev_fl=evoked_flf.info
evoked_fl2=mne.EvokedArray(np.array([fl_av]), info_ev_fl, tmin=-0.2)

fl_av=-np.squeeze(evoked_fl2.filter(l_freq=l_freq,h_freq=h_freq).get_data())


########################################################
####### changing now directory for saving plots

os.chdir(save_directory_A)

#############################################################
# tg=int(sfreq_g*2)
# tfl=int(sfreq_fl*2)
g_ev_data=np.squeeze(evoked_gf.get_data())
#aux_ev_data=np.squeeze(evoked_aux.get_data())*-1e-12
evoked_gf.times.shape

#m1=int(0.3*sfreq_g)

m1=int(0.3*sfreq_g)
m2=int(0.3*sfreq_fl)
sd_s=int(sfreq_g*3/5)
sd_f=int(sfreq_g+sfreq_g/5)

sdf_s=int(sfreq_fl*3/5)
sdf_f=int(sfreq_fl+sfreq_fl/5)

#plt.plot(evoked_gf.times[s:f],aux_ev_data[s:f])

fl_ev_data=np.squeeze(evoked_flf.get_data())
# fl_ev_data.shape#

# evoked_gf.times.shape

#plt.plot(evoked_flf.times,fl_ev_data,label = 'FL grad',color='blue')
#to plot the whole set
# x_g=data_g['time'].iloc[0:tg]
# y1=data_g['B_T_cal'].iloc[0:tg]

# x_fl=data_fl['time'].iloc[0:tfl]
# y2=data_fl['FL0102-BZ_CL'].iloc[0:tfl]

#plotting evoked response of our and fl in the same plot


g_s_dev="{:.2e}".format(np.std(g_ev_data[sd_s:sd_f]))
fl_s_dev="{:.2e}".format(np.std(fl_ev_data[sdf_s:sdf_f]))
textbox='lf='+str(l_freq)+', hf='+str(h_freq)+', g_sd='+str(g_s_dev)+', fl_sd='+str(fl_s_dev)+', Ng='+str(N_events)+', Nfl='+str(N_events_fl)
print(textbox)
#%%
fig_comparison, ax1 = plt.subplots()
ylim=[-1e-12,1e-12]
ylimg=[-0.5e-12,0.5e-12]
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Gradiometer [T]', color=color)
ax1.set_title('Evoked: '+plot_title)

#ax1.plot(evoked_aux.times, aux_ev_data,  color='grey', lw=0.5)
#ax1.plot(evoked_gf.times[s:f], sine_func(evoked_gf.times[s:f], params[0], params[1], params[2]), color='black')
ax1.plot(evoked_gf.times, g_av,   color=color, zorder=10)

ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=ylim)
ax1.text(-0.2, ylim[0]*0.95, textbox)
ax1.axvline(evoked_gf.times[m1], color="black", linestyle="--", lw=0.3)
ax1.fill_between(epochs_gf.times, g_av - g_sem, g_av + g_sem,    color=color, alpha=.2)
#ax1.axvline(evoked_gf.times[m2], color="black", linestyle="--", lw=0.3)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Fieldline [T]', color=color)  # we already handled the x-label with ax1
ax2.plot(evoked_flf.times, fl_av, color=color, lw=1,  zorder=0)
ax2.axvline(evoked_flf.times[m2], color="blue", linestyle="-", lw=0.3)
#ax2.axvline(evoked_flf.times[sdf_f], color="blue", linestyle="-", lw=0.3)
ax2.fill_between(epochs_flf.times, fl_av - fl_sem, fl_av + fl_sem,    color=color, alpha=.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=ylim)

fig_comparison.tight_layout()  # otherwise the right y-label is slightly clipped
fig_comparison.savefig(str(dir_name) + '_evoked.png')
#plt.legend()
plt.show()
#%%


# fig_epochs_g = epochs_gf.copy().plot_image(picks="mag", combine="mean", title=plot_title+' grad')
# fig_epochs_g.savefig(str(dir_name) + '_epochs_g.png')
# fig_epochs_fl=epochs_flf.plot_image(picks="mag", combine="mean", title=plot_title+' FL')
# fig_epochs_fl.savefig(str(dir_name) + '_epochs_fl.png')





# #FFT 


tmin = 0.
tmax = 1.
fmin = 1.
fmax = 50.

sfreq_gf = epochs_gf.info['sfreq']
sfreq_flf = epochs_flf.info['sfreq']
###############################
###### average of fft ################


spectrum_g = epochs_gf.compute_psd(
    'welch',
    n_fft=int(sfreq_gf * (tmax - tmin)),

    n_overlap=0, n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax,
    picks=['B_T_cal'],
    verbose=False)
psds_gf, freqs_gf = spectrum_g.get_data(return_freqs=True)

spectrum_flf = epochs_flf.compute_psd(
    'welch',
    n_fft=int(sfreq_flf * (tmax - tmin)),
    n_overlap=0, n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax,
    picks=picks_fl,
    verbose=False)
psds_flf, freqs_flf = spectrum_flf.get_data(return_freqs=True)


#************************
freq_range_g = range(np.where(np.floor(freqs_gf) == 1.)[0][0], np.where(np.ceil(freqs_gf) == fmax - 1)[0][0])
axis=(0,1)
psdsgf_plot = 10 * np.log10(psds_gf)
psdsgf_mean = psdsgf_plot.mean(axis=axis)[freq_range_g]
psdsgf_std = psdsgf_plot.std(axis=(0, 1))[freq_range_g]


freq_range_fl = range(np.where(np.floor(freqs_flf) == 1.)[0][0], np.where(np.ceil(freqs_flf) == fmax - 1)[0][0])
psdsfl_plot = 10 * np.log10(psds_flf)
psdsfl_mean = psdsfl_plot.mean(axis=(0, 1))[freq_range_fl]
psdsfl_std = psdsfl_plot.std(axis=(0, 1))[freq_range_fl]


#print(psdsgf_mean.shape, psdsfl_mean.shape)


fig_psd_comparison, ax1 = plt.subplots()
ylim=[-270,-240]

color = 'tab:red'
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Gradiometer [dB]', color=color)
ax1.set_title('PSD: '+plot_title)
ax1.plot(freqs_gf[freq_range_g], psdsgf_mean, color=color)
ax1.fill_between(freqs_gf[freq_range_g], psdsgf_mean - psdsgf_std, psdsgf_mean + psdsgf_std,
    color=color, alpha=.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=ylim)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Fieldline [dB]', color=color)  # we already handled the x-label with ax1
ax2.plot(freqs_flf[freq_range_fl], psdsfl_mean, color=color)
ax2.fill_between(freqs_flf[freq_range_fl], psdsfl_mean - psdsfl_std, psdsfl_mean + psdsfl_std,
    color=color, alpha=.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=ylim)

fig_psd_comparison.tight_layout()  # otherwise the right y-label is slightly clipped
fig_psd_comparison.savefig(str(dir_name) + '_psd.png')
plt.show()
print(dir_name +' plots saved')

#%%
######################### time frequency


freqs = np.arange(2, 25, 1)
n_cycles = freqs/2
time_bandwidth = 1.0



power_gf=mne.time_frequency.tfr_multitaper(evoked_gf, n_cycles=n_cycles, return_itc=False, freqs=freqs, decim=3)

fig_trf_g=power_gf.plot(picks=['B_T_cal'], title=plot_title+' TFR gradiometer')

fig_trf_g[0].savefig(str(dir_name) + '_trf_g.png')



power_flf = mne.time_frequency.tfr_multitaper(evoked_flf, n_cycles=n_cycles, return_itc=False, freqs=freqs, decim=3)
fig_trf_fl = power_flf.plot(picks_fl,title= plot_title+' TFR Fieldline')
fig_trf_fl[0].savefig(str(dir_name) + '_trf_fl.png')

