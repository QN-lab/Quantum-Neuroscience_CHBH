# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:53:11 2023

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
base_directory='Z:\\Data\\2023_07_31\\Z\\run\\grad_heart_6' # MAKE SURE THERE IS NO \\ AT THE END
os.chdir(base_directory)
if "C1" in base_directory:
    
    mode='Magnetometer'
elif 'c1' in base_directory:
    mode='Magnetometer'
else: 
    mode='Gradiometer'

mode_fl='Magnetometer'#### choose here####
#mode_fl='Gradiometer'    

print('Our sensor is in mode: '+mode +' \n'+ 'FL sensor is in mode: '+mode_fl)
P='50m'#dir_name[6:]
P1=50 #here in mili
BW='50'#dir_name[0:2]

#####################  ********** ########################

save_directory_A=base_directory[:19]+'\\Analysis' 

scal_fac=round(-299*P1**(-0.779),2)
print('P is '+str(P) +', BW is '+str(BW) +', SF is '+str(scal_fac))

dir_name = os.path.basename(base_directory)

plot_title='Heartbeat measurement'#AER '+str(mode)+' (SF='+str(scal_fac)+', P='+str(P)+'m, I=100u, BW='+str(BW)+'), ' +'\n'+str(dir_name)

##################### Raw - Our sensor ###########################
#%%
data_g=pd.read_csv('_g_r.csv',sep=',')
#print(data_g.head(3))
#data_g.shape
# tg=500000
# x=data_g['time'].iloc[0:tg]
# y=data_g['Demod_X'].iloc[0:tg]
# plt.plot(x,y)
# plt.show()
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

data_fl=pd.read_csv('_fl_r.csv',sep=',')
data_fl['time']=data_fl['time']
data_fl['Grad1']=data_fl['FL0101-BZ_CL']-data_fl['FL0102-BZ_CL']
#data_fl['Grad2']=data_fl['FL0103-BZ_CL']-data_fl['FL0104-BZ_CL']
print(data_fl.head(3))
sfreq_fl=1/(data_fl['time'].iloc[1])

ch_names_fl=data_fl.columns.tolist()

ch_types_fl=['mag']*len(ch_names_fl)
ch_types_fl[0]='misc'
ch_types_fl[-3:-1]=['stim']*2
ch_types_fl[-1]='grad'

print(ch_names_fl,len(ch_names_fl))
print(ch_types_fl,len(ch_types_fl))


data_raw_fl=data_fl.T

info_fl = mne.create_info(ch_names=ch_names_fl, ch_types=ch_types_fl, sfreq=sfreq_fl)
raw_fl= mne.io.RawArray(data_raw_fl, info_fl, verbose=True)


#%% Filtering
  
reject_criteria = dict(mag=19.5e-2)
reject_criteria_f = dict(mag=30e-2)
l_freq_r = 0.01 ##filter settings for the raw #(doesn't do much right now)
h_freq_r =20
l_freq = 0.01 ##fiter settings for evoked data
h_freq = 20
tmin_e = -0.2
tmax_e = 1
baseline =(-0.2,0)

if 'G' in mode_fl:
    picks_fl=['Grad1']
else:
    picks_fl = ['FL0101-BZ_CL']

  
events_g = mne.find_events(raw_g, stim_channel=['Stim'], min_duration=0.01, mask_type='not_and', \
                                     mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)

raw_gf = raw_g.copy().filter(l_freq=l_freq_r,h_freq=h_freq_r)

array_gf=raw_gf.get_data()
data_gf=pd.DataFrame(array_gf.T, columns=ch_names)

epochs_gf=mne.Epochs(raw_gf,picks=['mag'], events=events_g, event_id=3, tmin=tmin_e, tmax=tmax_e,baseline=baseline, reject=reject_criteria, preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)


events_m_fl = mne.find_events(raw_fl, stim_channel=['Stim'], min_duration=0.01,   mask_type='not_and', mask=(pow(2,8)+pow(2,9)+pow(2,10)+pow(2,11)+pow(2,12)+pow(2,13)),output='onset',consecutive=True)


raw_flf = raw_fl.copy().filter(l_freq=l_freq_r, h_freq=h_freq_r)

epochs_flf=mne.Epochs(raw_flf,events=events_m_fl, picks=picks_fl, tmin=tmin_e,tmax=tmax_e, baseline =baseline, reject=reject_criteria_f,preload=True)#.filter(l_freq=l_freq,h_freq=h_freq)

array_flf=raw_flf.get_data()
data_flf=pd.DataFrame(array_flf.T, columns=ch_names_fl)

#%% ############# run only once this part ( after running mne part)
st_ind=int(events_m_fl[0,0]+tmin_e*sfreq_fl)
end_ind=int(events_m_fl[-1,0]+tmax_e*sfreq_fl)
end_ind_g=int(events_g[-1,0]+tmax_e*sfreq_g)

data_flf=data_flf[st_ind:end_ind]
data_flf.shape
#%% crop both to the same length
tcrop=550
start=98
stop=start+5
data_gf=data_gf[0:int(tcrop*sfreq_g)]
data_flf=data_flf[0:int(tcrop*sfreq_fl)]
#%% long term plot
# x1=data_gf['time']
# y1=data_gf['B_T_cal']-data_gf.iloc[0][5]
# print(data_gf.head(1), data_flf.head(1))
# print(data_g.shape, data_g.shape)
# print(data_fl.shape, data_flf.shape)

# #y1=data_g['value']
# x2=data_fl['time']-data_fl.iloc[0][0]
# y2=data_fl['FL0101-BZ_CL']-data_fl.iloc[0][1]
# #y3=data_fl['FL0102-BZ_CL']-data_fl.iloc[0][2]
# #y4=data_fl['FL0103-BZ_CL']-data_fl.iloc[0][3]
# #y5=data_fl['FL0104-BZ_CL']-data_fl.iloc[0][4]
# #y6=-np.sqrt(y2**2+y4**2+y3**2)
# #data_fl['Grad1']=data_fl['FL0101-BZ_CL']-data_fl['FL0102-BZ_CL']
# #y2=data_fl['Grad1']
# textbox=''
os.chdir(origin_directory)

x1=data_gf['time']
y1=(data_gf['B_T_cal']-data_gf.iloc[0][5])*1
#y3=data_g['error_deg']*180/np.pi#-data_g.iloc[0][5]
#y1=data_g['value']
x2=data_flf['time']-data_flf.iloc[0][0]
y2=data_flf['FL0101-BZ_CL']-data_flf.iloc[0][1]

y3=data_flf['FL0102-BZ_CL']-data_flf.iloc[0][2]
y4=data_flf['FL0103-BZ_CL']-data_flf.iloc[0][3]
y5=data_flf['FL0104-BZ_CL']-data_flf.iloc[0][4]
#y6=-np.sqrt(y2**2+y4**2+y5**2)
y6=-y2-y4+y5

print(data_gf.head(2),'\n',data_flf.head(2))
#data_fl['Grad1']=data_fl['FL0101-BZ_CL']-data_fl['FL0102-BZ_CL']
#y2=data_fl['Grad1']
Export_all={'Grad time': x1.reset_index(drop=True), 'Grad B': y1.reset_index(drop=True), 'Ref time': x2.reset_index(drop=True), 'Ref Bz':  -y2.reset_index(drop=True),
            'Ref Bx':  y4.reset_index(drop=True),'Ref By':  -y5.reset_index(drop=True)}
export_all=pd.DataFrame(data=Export_all)
export_all.to_csv(str(dir_name) + '_start_'+str(start) +'_'+str(h_freq_r)+"_heart_all.csv", sep=',', index=False) 

#%%
fig_comparison2, ax1 = plt.subplots()
y2lim=[-0.5e-9,0.5e-9]
y1lim=[-2e-9,2e-9]

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel(mode +' [T]', color=color)
ax1.set_title('Heart measurement')#+dir_name)

#ax1.plot(evoked_aux.times, aux_ev_data,  color='grey', lw=0.5)
#ax1.plot(evoked_gf.times[s:f], sine_func(evoked_gf.times[s:f], params[0], params[1], params[2]), color='black')
ax1.plot(x1, -y1,   color=color, zorder=10)
#ax1.plot(x2, y2, color='blue', lw=1,  zorder=0)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=y2lim)
#ax1.text(-0.2, y2lim[0]*0.95, textbox)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Reference '+mode_fl+' [T]', color=color)  # we already handled the x-label with ax1
ax2.plot(x2, y2, color=color, lw=1,  zorder=0)
ax2.plot(x2, y3, color='purple', lw=1,  zorder=0)
ax2.plot(x2, -y4, color='cyan', lw=1,  zorder=0)
ax2.plot(x2, y5, color='orange', lw=1,  zorder=0)
ax2.plot(x2, y6, color='green', lw=1,  zorder=0)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=y1lim)

fig_comparison2.tight_layout()  # otherwise the right y-label is slightly clipped
fig_comparison2.savefig(str(dir_name) + '_long_run_comparison.png')
#plt.legend()
plt.show()
#%% Selection for the heartbeat data



fig_heart, ax1 = plt.subplots()
y1lim=[-0.15e-10,-0e-10]
y2lim=[0.02e-10,1.8e-10]
yscale=round((y1lim[1]-y2lim[0])/(y2lim[1]-y2lim[0]),2)

color = 'tab:red'
textbox='y2 scale:'+str(yscale)
ax1.set_xlabel('time (s)')
ax1.set_ylabel(mode +' [T]', color=color)
ax1.set_title('Lifts test '+dir_name)

#ax1.plot(evoked_aux.times, aux_ev_data,  color='grey', lw=0.5)
#ax1.plot(evoked_gf.times[s:f], sine_func(evoked_gf.times[s:f], params[0], params[1], params[2]), color='black')
ax1.plot(x1[int(start*sfreq_g):int(stop*sfreq_g)], y1[int(start*sfreq_g):int(stop*sfreq_g)],   color=color, lw=1, zorder=10)
#ax1.plot(x2, y2, color='blue', lw=1,  zorder=0)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=y1lim)
ax1.text(x1[int(start*sfreq_g)], y1lim[0]*0.9, textbox)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Reference field [T]', color=color)  # we already handled the x-label with ax1
ax2.plot(x2[int(start*sfreq_fl):int(stop*sfreq_fl)], y2[int(start*sfreq_fl):int(stop*sfreq_fl)], color=color, lw=1,  zorder=1)

ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=y2lim)

fig_heart.tight_layout()  # otherwise the right y-label is slightly clipped
fig_heart.savefig(str(dir_name) + '_start_'+str(start) +'_'+str(h_freq_r)+'_heart.png')
#plt.legend()
plt.show()

Export={'Grad time': x1[int(start*sfreq_g):int(stop*sfreq_g)].reset_index(drop=True), 
        'Grad B': y1[int(start*sfreq_g):int(stop*sfreq_g)].reset_index(drop=True), 
        'Ref time': x2[int(start*sfreq_fl):int(stop*sfreq_fl)].reset_index(drop=True), 
        'Ref Bz':  -y2[int(start*sfreq_fl):int(stop*sfreq_fl)].reset_index(drop=True),
        'Ref Bx':  y4[int(start*sfreq_fl):int(stop*sfreq_fl)].reset_index(drop=True),
        'Ref By':  -y5[int(start*sfreq_fl):int(stop*sfreq_fl)].reset_index(drop=True)}
export=pd.DataFrame(data=Export)
export.to_csv(str(dir_name) + '_start_'+str(start) +'_'+str(h_freq_r)+"_heart_slice.csv", sep=',', index=False)                                                                                                                                                    

 #%% other columns
# x1=data_g['time']
# y1=data_g['B_T_cal']-data_g.iloc[0][4]
# y3=data_g['error_deg']*180/np.pi#-data_g.iloc[0][5]
# #y1=data_g['value']
# x2=data_fl['time']-data_fl.iloc[0][0]
# y2=data_fl['FL0101-BZ_CL']-data_fl.iloc[0][1]
# data_fl['Grad1']=data_fl['FL0101-BZ_CL']-data_fl['FL0102-BZ_CL']
# #y2=data_fl['Grad1']
# textbox=''

# fig_comparison2, ax1 = plt.subplots()
# y2lim=[-2e-9,1e-9]

# y1lim=[-10,10]
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('Gradiometer [T]', color=color)
# ax1.set_title('Lifts test '+dir_name)

# #ax1.plot(evoked_aux.times, aux_ev_data,  color='grey', lw=0.5)
# #ax1.plot(evoked_gf.times[s:f], sine_func(evoked_gf.times[s:f], params[0], params[1], params[2]), color='black')
# ax1.plot(x1, y1,   color=color, zorder=10)
# ax1.plot(x2, y2, color='blue', lw=1,  zorder=0)
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set(ylim=y2lim)
# ax1.text(-0.2, y2lim[0]*0.95, textbox)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('Fieldline [T]', color=color)  # we already handled the x-label with ax1
# ax2.plot(x1, y3, color='green', lw=1,  zorder=0)

# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set(ylim=y1lim)

# fig_comparison2.tight_layout()  # otherwise the right y-label is slightly clipped
# fig_comparison2.savefig(str(dir_name) + '_long_run_comparison.png')
# #plt.legend()
# plt.show()