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
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.get_option("display.max_columns",15)

#Z:\Data\Processing_directory\Attenuation_gradiometer
base_directory='Z:\\Data\\2023_06_09_brain\\Z\\sig\\brain6_25BW_700'


dir_name = os.path.basename(base_directory)
os.chdir(base_directory)
P='50m'#dir_name[6:]
P1=50 #here in mili
BW='25'#dir_name[0:2]

#plots will be saved in
base_directory_A='Z:\\Data\\2023_06_09_brain\\Analysis\\' #################### here!!!!!!!!!!!!!!!!!!!!!

scal_fac=round(-299*P1**(-0.779),2)
print('P is '+str(P) +', BW is '+str(BW) +', SF is '+str(scal_fac))
file_title=str(base_directory)[8:]


plot_title='Brain Grad'+' (SF='+str(scal_fac)+', P='+str(P)+'m, I=2m, BW='+str(BW)+'), ' +'\n'+str(dir_name)


##################### Raw - Our sensor ###########################

data_g=pd.read_csv('_g_r.csv',sep=',')
#print(data_g.head(3))
#data_g.shape

scal_fac=scal_fac#PLL scaling factor

data_g['B_T_cal']=data_g['B_T (pT)']*scal_fac*1e-12
data_g['Trig_in2']=data_g['Trig_in2'].round(0).astype(int)
data_g['Aux1_v']=data_g['Aux1_v'].round(0).astype(int)
pd.set_option('display.max_columns', 20)
print(data_g.head(1))

ch_names = ['chunk', 'value', 'time', 'B_T (pT)','error_deg', 'Aux1_v', 'Aux_2', 'Trig_in2','Demod_X', 'Demod_Y', 'Stim','B_T_cal']

ch_types_g = ['misc', 'misc', 'misc','misc', 'misc','eeg','stim', 'stim', 'misc', 'misc', 'stim','mag']
data_raw_g=data_g.T

sfreq_g=1/data_g['time'].iloc[1]
print(sfreq_g)
info_g = mne.create_info(ch_names=ch_names, ch_types=ch_types_g, sfreq=sfreq_g)
raw_g = mne.io.RawArray(data_raw_g, info_g, verbose=True)


###################################################################
### Raw - fieldline 
base_directory_fl='Z:\\Data\\2023_04_26_FL_Brain'
dir_name_fl = os.path.basename(base_directory_fl)
os.chdir(base_directory_fl)

fl_file_name='20230426_Ania_brain_left_raw.fif'
raw_fl=mne.io.read_raw_fif(fl_file_name,preload=True)
info = mne.io.read_info(fl_file_name)
print(info)

data_fl=raw_fl.to_data_frame()
#print(data_fl.head(3))
#change units to T
data_fl.iloc[:,1:5]=data_fl.iloc[:,1:5]*1e-15
#Analogue input on Fieldline makes it difficult for MNE to find the events so we round the values to the nearest integer 
data_fl['Input1']=data_fl['Input1'].round(0).astype(int)
data_fl['Input1']=data_fl['Input1'].replace(1,2,inplace=False)
print(data_fl.head(1))

ch_names_fl = ['time', 'FL0101-BZ_CL','FL0102-BZ_CL','FL0103-BZ_CL',  'FL0104-BZ_CL','Input1']
ch_types_fl = ['misc',  'mag', 'mag','mag','mag', 'stim']

raw_data_fl=data_fl.T

sfreq_fl =  1000.0 #in Hz


#df0.plot(x='time', y='FL0103-BZ_CL'),#ylim=(-1e-11,+1e-11))
# plt.show()

info_fl = mne.create_info(ch_names=ch_names_fl, ch_types=ch_types_fl, sfreq=sfreq_fl)
raw_fl= mne.io.RawArray(raw_data_fl, info_fl, verbose=True)


############### events epoching and rejection

reject_criteria = dict(mag=10e-10)
l_freq_r = 0.01
h_freq_r = 200
l_freq = 0.01
h_freq = 20
tmin_e = -0.2
tmax_e = 1.8
baseline =(-0.2,0)
events_g = mne.find_events(raw_g, stim_channel=['Stim'], min_duration=0.01, mask_type='not_and', \
                                     mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)
#mne.viz.plot_events(events_all_g,first_samp=3000 ) 

N_events=events_g.shape[0]

raw_gf = raw_g.copy().filter(l_freq=l_freq_r,h_freq=h_freq_r)


epochs_gf=mne.Epochs(raw_gf,picks=['mag'], events=events_g, event_id=3, tmin=tmin_e, tmax=tmax_e,baseline=baseline, reject=reject_criteria, preload=True).filter(l_freq=l_freq,h_freq=h_freq)
evoked_gf=epochs_gf.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)
#evoked_gf.plot()

#plot  to check if the stim chanell is rejecting trials with the temperatrue sensing on
    #raw_g.copy().pick_types(meg=True, stim=True).plot(events=events_g, start=0, duration=20, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'), event_color={ 3:'b'})
 
epochs_aux=mne.Epochs(raw_gf, picks=['eeg'], events=events_g, event_id=3, tmin=tmin_e, tmax=tmax_e,baseline=(-0.2,0),  preload=True).filter(l_freq=l_freq,h_freq=h_freq)
evoked_aux=epochs_aux.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)

### FL 

events_all_fl = mne.find_events(raw_fl, stim_channel=['Input1'], min_duration=0.01, mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)
#mne.viz.plot_events(events_all_fl) 

# epochs_all_fl=mne.Epochs(raw_fl,events=events_all_fl, event_id=2, tmin=-0.2,tmax=1, baseline=(0.6,0.8),  preload=True)
# evoked_all_fl=epochs_all_fl.average()
# evoked_all_fl.plot(spatial_colors=True)

raw_flf = raw_fl.copy().filter(l_freq=l_freq_r, h_freq=h_freq_r)

reject_criteria_f = dict(mag=1e-10)
epochs_flf=mne.Epochs(raw_flf,events=events_all_fl[:N_events], tmin=tmin_e,tmax=tmax_e, baseline =(-0.2,0), reject=reject_criteria_f,preload=True).filter(l_freq=l_freq,h_freq=h_freq)
evoked_flf=epochs_flf.copy().average(method='mean').filter(l_freq=l_freq,h_freq=h_freq)
#evoked_flf.plot()

# #evoked_fl_exp = evoked_flf.copy().pick_types(meg=True,stim=False)


###### plotting


# raw_fl.copy().pick_types(meg=True, stim=True).plot(events=events_fl, start=50, duration=50, scalings=dict(mag=1e-10),  color=dict(mag='darkblue', stim='gray'))

# tg=int(sfreq_g*2)
# tfl=int(sfreq_fl*2)
g_ev_data=np.squeeze(evoked_gf.get_data())
aux_ev_data=np.squeeze(evoked_aux.get_data())*-1e-12
#evoked_gf.times.shape
s=170
f=670
sd_s=650
sd_f=sd_s+int(sfreq_g)

sdf_s=775
sdf_f=1775

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
from scipy import optimize

def sine_func(x, a, b,c):
    return a * -np.sin(b * x)+c

params, params_covariance = optimize.curve_fit(sine_func, evoked_gf.times[s:f], g_ev_data[s:f],
                                               p0=[1e-12, 30,1e-11])

print(params)

g_s_dev="{:.2e}".format(np.std(g_ev_data[sd_s:sd_f]))
fl_s_dev="{:.2e}".format(np.std(fl_ev_data[sdf_s:sdf_f]))
textbox='lp='+str(l_freq)+', hp='+str(h_freq)+', g_sdev='+str(g_s_dev)+', fl_sdev='+str(fl_s_dev)+', N='+str(N_events)
print(textbox)

fig_comparison, ax1 = plt.subplots()
ylim=[-2e-12,2e-12]
color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('Magnetometer [T]', color=color)
ax1.set_title('Evoked: '+plot_title)

ax1.plot(evoked_aux.times, aux_ev_data,  color='grey', lw=0.5)
#ax1.plot(evoked_gf.times[s:f], sine_func(evoked_gf.times[s:f], params[0], params[1], params[2]), color='black')
ax1.plot(evoked_gf.times, g_ev_data,   color=color, zorder=10)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=ylim)
ax1.text(-0.2, ylim[0]*0.95, textbox)
ax1.axvline(evoked_gf.times[sd_s], color="black", linestyle="--", lw=0.3)
ax1.axvline(evoked_gf.times[sd_f], color="black", linestyle="--", lw=0.3)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Fieldline [T]', color=color)  # we already handled the x-label with ax1
ax2.plot(evoked_flf.times, fl_ev_data, color=color, lw=1,  zorder=0)
ax2.axvline(evoked_flf.times[sdf_s], color="blue", linestyle="-", lw=0.3)
ax2.axvline(evoked_flf.times[sdf_f], color="blue", linestyle="-", lw=0.3)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=ylim)

fig_comparison.tight_layout()  # otherwise the right y-label is slightly clipped
fig_comparison.savefig(str(dir_name) + 'all_evoked.png')
plt.legend()
plt.show()


#### amplitude fitting



######################### time frequency


# freqs = np.arange(2, 25, 1)
# n_cycles = freqs/2
# time_bandwidth = 3.0


# power_flf = mne.time_frequency.tfr_multitaper(evoked_flf, n_cycles=n_cycles, return_itc=False, freqs=freqs, decim=3)
# power_flf.plot(['FL0103-BZ_CL'],title='TFR Fieldline')

# power_gf = mne.time_frequency.tfr_multitaper(evoked_gf, n_cycles=n_cycles, return_itc=False, freqs=freqs, decim=3)
# power_gf.plot(['B_T_cal'],title='TFR our gradiometer')

#FFT 


tmin = 0.
tmax = 1.
fmin = 1.
fmax = 65.

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
    n_fft=int(sfreq_gf * (tmax - tmin)),
    n_overlap=0, n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax,
    picks=['FL0103-BZ_CL'],
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
ylim=[-300,-200]
color = 'tab:red'
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Magnetometer [T]', color=color)
ax1.set_title('PSD: '+plot_title)
ax1.plot(freqs_gf[freq_range_g], psdsgf_mean, color=color)
ax1.fill_between(freqs_gf[freq_range_g], psdsgf_mean - psdsgf_std, psdsgf_mean + psdsgf_std,
    color=color, alpha=.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=ylim)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Fieldline [T]', color=color)  # we already handled the x-label with ax1
ax2.plot(freqs_flf[freq_range_fl], psdsfl_mean, color=color)
ax2.fill_between(freqs_flf[freq_range_fl], psdsfl_mean - psdsfl_std, psdsfl_mean + psdsfl_std,
    color=color, alpha=.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=ylim)

fig_psd_comparison.tight_layout()  # otherwise the right y-label is slightly clipped
fig_psd_comparison.savefig(str(dir_name) + '_all_psd.png')
plt.show()
