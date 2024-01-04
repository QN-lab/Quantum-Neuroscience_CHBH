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
mag_directory='Z:\\Data\\2023_08_17_bench\\mag_noise\\mag_noise_21mV_5pTcm_000' # MAKE SURE THERE IS NO \\ AT THE END
grad_directory='Z:\\Data\\2023_08_17_bench\\grad_noise\\grad_noise_21mV_5pTcm_000' # MAKE SURE THERE IS NO \\ AT THE END
save_directory='Z:\\Data\\2023_08_17_bench\\Analysis\\'



mode='Gradiometer vs Magnetometer'

P='50m'#dir_name[6:]
P1=50 #here in mili
BW='100'#dir_name[0:2]

#####################  ********** ########################


scal_fac=round(-299*P1**(-0.779),2)
print('P is '+str(P) +', BW is '+str(BW) +', SF is '+str(scal_fac))

dir_name = os.path.basename(mag_directory)[4:-4]
print(dir_name)
plot_title='BW='+str(BW)+'), ' +'\n'+str(dir_name)

##################### Raw - Our sensor ###########################
#%%
os.chdir(grad_directory)
data_g=pd.read_csv('_f.csv',sep=',')
#print(data_g.head(3))
#data_g.shape

data_g['B_T_cal']=data_g['B_T (pT)']*scal_fac*1e-12
data_g['Trig_in2']=data_g['Trig_in2'].round(0).astype(int)
data_g['Aux1_v']=data_g['Aux1_v'].round(0).astype(int)

print(data_g.head(1))
#data_g.to_csv("_g_r_raw.csv", sep=',', index=False)
data_g=data_g.drop(data_g.columns[[0,1,3,4,8,9]], axis=1, inplace=False)
print(data_g.head(1))                                    
ch_names=data_g.columns.tolist()
ch_types=['misc']*len(ch_names)
ch_types[2]='stim'
ch_types[4]='stim'
ch_types[-1]='mag'
#ch_names = ['chunk', 'value', 'time', 'B_T (pT)','error_deg', 'Aux1_v', 'Aux_2', 'Trig_in2','Demod_X', 'Demod_Y', 'Stim','B_T_cal']
#ch_types_g = ['misc', 'misc', 'misc','misc', 'misc','eeg','stim', 'stim', 'misc', 'misc', 'stim','mag']

sfreq_g=1/data_g['time'].iloc[1]
data_raw_g=data_g.T
print(sfreq_g)
info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq_g)
raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)

######################### Raw - fieldline #############################################
os.chdir(mag_directory)

data_m=pd.read_csv('_f.csv',sep=',')
#print(data_m.head(3))
#data_m.shape

data_m['B_T_cal']=data_m['B_T (pT)']*scal_fac*1e-12
data_m['Trig_in2']=data_m['Trig_in2'].round(0).astype(int)
data_m['Aux1_v']=data_m['Aux1_v'].round(0).astype(int)
x=data_m[ch_names[0]]
y1=data_m[ch_names[3]]
y2=data_m[ch_names[3]]

data_m.plot(x=ch_names[0],y=ch_names[2:6])
plt.show()
print(data_m.head(1))
#data_m.to_csv("_m_r_raw.csv", sep=',', index=False)
data_m=data_m.drop(data_m.columns[[0,1,3,4,8,9]], axis=1, inplace=False)
print(data_m.head(1))                                    
ch_names=data_m.columns.tolist()
ch_types=['misc']*len(ch_names)
ch_types[2]='stim'
ch_types[4]='stim'
ch_types[-1]='mag'
#ch_names = ['chunk', 'value', 'time', 'B_T (pT)','error_deg', 'Aux1_v', 'Aux_2', 'Trig_in2','Demod_X', 'Demod_Y', 'Stim','B_T_cal']
#ch_types_m = ['misc', 'misc', 'misc','misc', 'misc','eeg','stim', 'stim', 'misc', 'misc', 'stim','mag']

sfreq_m=1/data_m['time'].iloc[1]
data_raw_m=data_m.T
print(sfreq_m)
info_m = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq_m)
raw_m = mne.io.RawArray(data_raw_m, info_m, verbose=True)
#%% ############### events epoching and rejection
os.chdir(save_directory)
reject_criteria = dict(mag=19.5e-7)

l_freq_r = 0.001 ##filter settings for the raw #(doesn't do much right now)
h_freq_r =400
l_freq = 0.001 ##fiter settings for evoked data
h_freq = 400
tmin_e = -0.2
tmax_e = 2.5
baseline =(-0.2,0)

    
events_g = mne.find_events(raw_g, stim_channel=['Stim'], min_duration=0.01, mask_type='not_and', \
                                     mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step',consecutive=True)
# events_g0 = mne.find_events(raw_g, stim_channel=['Stim'], min_duration=0.01, mask_type='not_and', \
#                                      mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='offset',consecutive=True)

mne.viz.plot_events(events_g) 

raw_gf = raw_g.copy().filter(l_freq=l_freq_r,h_freq=h_freq_r)


epochs_gnf=mne.Epochs(raw_gf,picks=['mag'], events=events_g, event_id=3, tmin=tmin_e, tmax=tmax_e,baseline=baseline, reject=reject_criteria, preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)
evoked_gnf=epochs_gnf.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)
epochs_g0f=mne.Epochs(raw_gf,picks=['mag'], events=events_g, event_id=0, tmin=tmin_e, tmax=tmax_e,baseline=baseline, reject=reject_criteria, preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)
evoked_g0f=epochs_gnf.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)

N_events=len(epochs_gnf)#or events_gf.shape[0]
print(N_events)
#evoked_gf.plot()

#plot  to check if the stim chanell is rejecting trials with the temperatrue sensing on
raw_g.copy().pick_types(meg=True, stim=True).plot(events=events_g, start=0, duration=30, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'))
 

# events_all_m = mne.find_events(raw_m, stim_channel=['Input1'], min_duration=0.01,  mask_type='not_and', \
#                                 mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)
# # matched events to our sensor
events_m = mne.find_events(raw_m, stim_channel=['Stim'], min_duration=0.01,   mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='step',consecutive=True)

raw_m.copy().pick_types(meg=True, stim=True).plot(events=events_m, start=0, duration=25, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'))
#mne.viz.plot_events(events_m_m) 


raw_mf = raw_m.copy().filter(l_freq=l_freq_r, h_freq=h_freq_r)

epochs_mnf=mne.Epochs(raw_mf,events=events_m, picks=['mag'], event_id=3, tmin=tmin_e,tmax=tmax_e, baseline =baseline, reject=reject_criteria,preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)
evoked_mf=epochs_mnf.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)
epochs_m0f=mne.Epochs(raw_mf,events=events_m, picks=['mag'], event_id=0, tmin=tmin_e,tmax=tmax_e, baseline =baseline, reject=reject_criteria,preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)
evoked_m0f=epochs_mnf.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)

N_events_m = len(epochs_mnf)#events_m_m.shape[0]

np.shape(evoked_mf)
#
#evoked_mf.plot()

# #evoked_m_exp = evoked_mf.copy().pick_types(meg=True,stim=False)

#%%
###### plotting
gn_epochs_data=np.squeeze(epochs_gnf.get_data())
gn_epochs_DF=pd.DataFrame(data=(gn_epochs_data.T))

#g_epochs_DF.insert(0, 'time', epochs_gf.times)
#g_epochs_DF.to_csv(str(dir_name)+"_origin.csv", sep=',', index=False)
#g_epochs_DF.plot(x='time', legend=None)

gn_epochs_DF.iloc[:,:].plot(legend=None, ylim=[-2e-11,2e-11],title='our sensor'+str(mode) )
plt.show()
#g_epochs_DF.to_csv("epochs_origin.csv", sep=',', index=False)
#g_av=g_epochs_DF.iloc[:,350:].mean(axis=1)

gn_av=g_epochs_DF.mean(axis=1)
gn_std=g_epochs_DF.std(axis=1)
gn_sem=g_epochs_DF.sem(axis=1)
gn_epochs_DF.shape                               
g_epochs_DF.head(3)

evoked_gf=epochs_gf.copy().average()
evoked_gf.shapeN_events=len(epochs_gf)#or events_gf.shape[0]
print(epochs_gf)

info_ev=evoked_gf.info
# evoked_g2=mne.EvokedArray(np.array([g_av]), info_ev, tmin=-0.2)

# g_av=np.squeeze(evoked_g2.filter(l_freq=l_freq,h_freq=h_freq).get_data())


mn_epochs_data=np.squeeze(epochs_mnf.get_data())
mn_epochs_DF=pd.DataFrame(data=(mn_epochs_data[0:].T))
# g_epochs_DF.insert(0, 'time', epochs_gf.times)
fl_av=fl_epochs_DF.mean(axis=1)
fl_std=fl_epochs_DF.std(axis=1)
fl_sem=fl_epochs_DF.sem(axis=1)
# raw_m.copy().pick_types(meg=True, stim=True).plot(events=events_m, start=50, duration=50, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'))

fl_epochs_DF.iloc[:,:].plot(legend=None, ylim=[-2e-11,2e-11],title='FL '+str(mode_m))
plt.show()
info_ev_m=evoked_mf.info
evoked_m2=mne.EvokedArray(np.array([fl_av]), info_ev_m, tmin=-0.2)

fl_av=-np.squeeze(evoked_m2.filter(l_freq=l_freq,h_freq=h_freq).get_data())

#%%
########################################################
####### changing now directory for saving plots

os.chdir(save_directory_A)

#############################################################
# tg=int(sfreq_g*2)
# tfl=int(sfreq_m*2)
g_ev_data=np.squeeze(evoked_gf.get_data())
#aux_ev_data=np.squeeze(evoked_aux.get_data())*-1e-12
evoked_gf.times.shape

#m1=int(0.3*sfreq_g)

m1=int(0.2*sfreq_g)
m2=int(0.3*sfreq_m)
sd_s=int(sfreq_g*3/5)
sd_f=int(sfreq_g+sfreq_g/5)

sdf_s=int(sfreq_m*3/5)
sdf_f=int(sfreq_m+sfreq_m/5)

#plt.plot(evoked_gf.times[s:f],aux_ev_data[s:f])

fl_ev_data=np.squeeze(evoked_mf.get_data())
# fl_ev_data.shape#

# evoked_gf.times.shape

#plt.plot(evoked_mf.times,fl_ev_data,label = 'FL grad',color='blue')
#to plot the whole set
# x_g=data_g['time'].iloc[0:tg]
# y1=data_g['B_T_cal'].iloc[0:tg]

# x_m=data_m['time'].iloc[0:tfl]
# y2=data_m['FL0102-BZ_CL'].iloc[0:tfl]

#plotting evoked response of our and fl in the same plot


g_s_dev="{:.2e}".format(np.std(g_ev_data[sd_s:sd_f]))
fl_s_dev="{:.2e}".format(np.std(fl_ev_data[sdf_s:sdf_f]))
textbox='lf='+str(l_freq)+', hf='+str(h_freq)+', g_sd='+str(g_s_dev)+', fl_sd='+str(fl_s_dev)+', Ng='+str(N_events)+', Nfl='+str(N_events_m)
print(textbox)
 #%%
# fig_comparison, ax1 = plt.subplots()
# ylim=[-1e-12,1e-12]
# ylimg=[-0.5e-12,0.5e-12]
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel(mode+' [T]', color=color)
# ax1.set_title('Evoked: '+plot_title)

# #ax1.plot(evoked_aux.times, aux_ev_data,  color='grey', lw=0.5)
# #ax1.plot(evoked_gf.times[s:f], sine_func(evoked_gf.times[s:f], params[0], params[1], params[2]), color='black')
# ax1.plot(evoked_gf.times, g_av,   color=color, zorder=10)

# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set(ylim=ylim)
# ax1.text(-0.2, ylim[0]*0.95, textbox)
# ax1.axvline(evoked_gf.times[m1], color="black", linestyle="--", lw=0.3, alpha=0.2)
# ax1.fill_between(epochs_gf.times, g_av - g_sem, g_av + g_sem,    color=color, alpha=.2)
# #ax1.axvline(evoked_gf.times[m2], color="black", linestyle="--", lw=0.3)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('FL '+mode_m+' [T]', color=color)  # we already handled the x-label with ax1
# ax2.plot(evoked_mf.times, fl_av, color=color, lw=1,  zorder=0)
# ax2.axvline(evoked_mf.times[m2], color="blue", linestyle="-", lw=0.3)
# #ax2.axvline(evoked_mf.times[sdf_f], color="blue", linestyle="-", lw=0.3)
# ax2.fill_between(epochs_mf.times, fl_av - fl_sem, fl_av + fl_sem,    color=color, alpha=.2)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set(ylim=ylim)

# fig_comparison.tight_layout()  # otherwise the right y-label is slightly clipped
# fig_comparison.savefig(str(dir_name) + '_evoked.png')
# #plt.legend()
# plt.show()
#%%


# fig_epochs_g = epochs_gf.copy().plot_image(picks="mag", combine="mean", title=plot_title+' grad')
# fig_epochs_g.savefig(str(dir_name) + '_epochs_g.png')
# fig_epochs_m=epochs_mf.plot_image(picks="mag", combine="mean", title=plot_title+' FL')
# fig_epochs_m.savefig(str(dir_name) + '_epochs_m.png')





# #FFT 


tmin = 0.
tmax = 2.
fmin = 0.001
fmax = 400.

sfreq_gf = epochs_gnf.info['sfreq']
sfreq_mf = epochs_mnf.info['sfreq']
###############################
###### average of fft ################


spectrum_gnf = epochs_gnf.compute_psd(
    'welch',
    n_fft=int(sfreq_gf * (tmax - tmin)),

    n_overlap=0, n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax,
    picks=['B_T_cal'],
    verbose=False)
psds_gnf, freqs_gnf = spectrum_gnf.get_data(return_freqs=True)


spectrum_g0f = epochs_g0f.compute_psd(
    'welch',
    n_fft=int(sfreq_gf * (tmax - tmin)),

    n_overlap=0, n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax,
    picks=['B_T_cal'],
    verbose=False)
psds_g0f, freqs_g0f = spectrum_g0f.get_data(return_freqs=True)

spectrum_mnf = epochs_mnf.compute_psd(
    'welch',
    n_fft=int(sfreq_mf * (tmax - tmin)),
    n_overlap=0, n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax,
    picks=['B_T_cal'],
    verbose=False)
psds_mnf, freqs_mnf = spectrum_mnf.get_data(return_freqs=True)

spectrum_m0f = epochs_m0f.compute_psd(
    'welch',
    n_fft=int(sfreq_mf * (tmax - tmin)),
    n_overlap=0, n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax,
    picks=['B_T_cal'],
    verbose=False)
psds_m0f, freqs_m0f = spectrum_m0f.get_data(return_freqs=True)

#************************
freq_range_g = range(np.where(np.floor(freqs_gnf) == 1.)[0][0], np.where(np.ceil(freqs_gnf) == fmax - 1)[0][0])
axis=(0,1)
psdsgnf_plot = 10 * np.log10(psds_gnf)
psdsgnf_mean = psdsgnf_plot.mean(axis=axis)[freq_range_g]
psdsgnf_std = psdsgnf_plot.std(axis=(0, 1))[freq_range_g]

freq_range_g0 = range(np.where(np.floor(freqs_g0f) == 1.)[0][0], np.where(np.ceil(freqs_g0f) == fmax - 1)[0][0])
axis=(0,1)
psdsg0f_plot = 10 * np.log10(psds_g0f)
psdsg0f_mean = psdsg0f_plot.mean(axis=axis)[freq_range_g0]
psdsg0f_std = psdsg0f_plot.std(axis=(0, 1))[freq_range_g0]

freq_range_m = range(np.where(np.floor(freqs_mnf) == 1.)[0][0], np.where(np.ceil(freqs_mnf) == fmax - 1)[0][0])
psdsmnf_plot = 10 * np.log10(psds_mnf)
psdsmnf_mean = psdsmnf_plot.mean(axis=(0, 1))[freq_range_m]
psdsmnf_std = psdsmnf_plot.std(axis=(0, 1))[freq_range_m]


freq_range_m0 = range(np.where(np.floor(freqs_mnf) == 1.)[0][0], np.where(np.ceil(freqs_mnf) == fmax - 1)[0][0])
psdsm0f_plot = 10 * np.log10(psds_mnf)
psdsm0f_mean = psdsm0f_plot.mean(axis=(0, 1))[freq_range_m0]
psdsm0f_std = psdsm0f_plot.std(axis=(0, 1))[freq_range_m0]
#print(psdsgf_mean.shape, psdsfl_mean.shape)

fft_gnf=np.sqrt(psds_gnf)
psds_gnf_mean = fft_gnf.mean(axis=axis)[freq_range_g]
psds_gnf_std = fft_gnf.std(axis=(0, 1))[freq_range_g]

fft_g0f=np.sqrt(psds_g0f)
psds_g0f_mean = fft_g0f.mean(axis=axis)[freq_range_g0]
psds_g0f_std = fft_g0f.std(axis=(0, 1))[freq_range_g0]

fft_mnf=np.sqrt(psds_mnf)
psds_mnf_mean = fft_mnf.mean(axis=(0, 1))[freq_range_m]
psds_mnf_std = fft_mnf.std(axis=(0, 1))[freq_range_m]

fft_m0f=np.sqrt(psds_m0f)
psds_m0f_mean = fft_m0f.mean(axis=(0, 1))[freq_range_m0]
psds_m0f_std = fft_m0f.std(axis=(0, 1))[freq_range_m0]

#print(psdsgf_mean.shape, psdsfl_mean.shape)
#%%

fig_fft_noise, ax1 = plt.subplots()
ylim=[1e-14,10e-11]
color = 'tab:red'
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Grad FFT fT/sqrt(Hz)', color=color)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.fill
ax1.set_title('FFT with noise applied' +str(dir_name))
ax1.plot(freqs_gnf[freq_range_g], psds_gnf_mean[freq_range_g], color=color)
ax1.plot(freqs_g0f[freq_range_g0], psds_g0f_mean[freq_range_g0], color='darkred')
#ax1.fill_between(freqs_gf[freq_range_g], psds_gf_mean - psds_gf_std, psds_gf_mean + psds_gf_std,     color=color, alpha=.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=ylim)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Mag FFT fT/sqrt(Hz)', color=color)  # we already handled the x-label with ax1
ax2.set_yscale('log')
ax2.plot(freqs_mnf[freq_range_m], psds_mnf_mean, color=color)
ax2.plot(freqs_m0f[freq_range_m0], psds_m0f_mean, color='darkblue')
#ax2.fill_between(freqs_mf[freq_range_m], psds_m_mean - psds_m_std, psds_m_mean + psds_m_std,     color=color, alpha=.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=ylim)
ax2.set_xscale('log')
fig_fft_noise.tight_layout()  # otherwise the right y-label is slightly clipped
fig_fft_noise.savefig(str(dir_name) + '_noise_fft.png')
plt.show()


fig_fft_no_noise, ax1 = plt.subplots()
ylim=[1e-15,10e-12]
color = 'tab:red'
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Grad FFT fT/sqrt(Hz)', color=color)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.fill
ax1.set_title('FFT with noise applied' +str(dir_name))
ax1.plot(freqs_g0f[freq_range_g0], psds_g0f_mean[freq_range_g0], color=color)
#ax1.fill_between(freqs_gf[freq_range_g], psds_gf_mean - psds_gf_std, psds_gf_mean + psds_gf_std,     color=color, alpha=.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=ylim)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Mag FFT fT/sqrt(Hz)', color=color)  # we already handled the x-label with ax1
ax2.set_yscale('log')
ax2.plot(freqs_m0f[freq_range_m0], psds_m0f_mean[freq_range_g0], color=color)
#ax2.fill_between(freqs_mf[freq_range_m], psds_m_mean - psds_m_std, psds_m_mean + psds_m_std,     color=color, alpha=.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=ylim)
ax2.set_xscale('log')
fig_fft_no_noise.tight_layout()  # otherwise the right y-label is slightly clipped
fig_fft_no_noise.savefig(str(dir_name) + '_no_noise_fft.png')
plt.show()






fig_psd_comparison, ax1 = plt.subplots()
ylim=[-270,-240]

color = 'tab:red'
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel(mode+' [dB]', color=color)
ax1.set_title('PSD: '+plot_title)
ax1.plot(freqs_gf[freq_range_g], psdsgf_mean, color=color, alpha=0.2)
ax1.fill_between(freqs_gf[freq_range_g], psdsgf_mean - psdsgf_std, psdsgf_mean + psdsgf_std,
    color=color, alpha=.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=ylim)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('FL '+mode_m+' [dB]', color=color)  # we already handled the x-label with ax1
ax2.plot(freqs_mf[freq_range_m], psdsfl_mean, color=color)
ax2.fill_between(freqs_mf[freq_range_m], psdsfl_mean - psdsfl_std, psdsfl_mean + psdsfl_std,
    color=color, alpha=.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=ylim)

fig_psd_comparison.tight_layout()  # otherwise the right y-label is slightly clipped
fig_psd_comparison.savefig(str(dir_name) + '_psd.png')
plt.show()
print(dir_name +' plots saved')

#%%
######################### time frequency


freqs = np.arange(1, 35, 1)
n_cycles = freqs/2
time_bandwidth = 1.0



power_gf=mne.time_frequency.tfr_multitaper(evoked_gf, n_cycles=n_cycles, return_itc=False, freqs=freqs, decim=3)

fig_trf_g=power_gf.plot(picks=['B_T_cal'], title=plot_title+' TFR '+mode)

fig_trf_g[0].savefig(str(dir_name) + '_trf_g.png')


power_mf = mne.time_frequency.tfr_multitaper(evoked_mf, n_cycles=n_cycles, return_itc=False, freqs=freqs, decim=3)
fig_trf_m = power_mf.plot(picks_m,title= plot_title+' TFR FL '+mode_m)
fig_trf_m[0].savefig(str(dir_name) + '_trf_m.png')

