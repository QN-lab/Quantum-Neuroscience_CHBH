# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:18:18 2023

@author: kowalcau
"""


data_flc=data_fl.copy()

data_gc=data_g.copy()
#%%long term 4 sensors 2 orthogonal

x1=data_g['time']
y0=data_g['B_T_cal']-data_g.iloc[0][5]


#y1=data_g['value']
x2=data_fl['time']-data_fl.iloc[0][0]
y1=data_fl['FL0101-BZ_CL']-data_fl.iloc[0][1]
y2=data_fl['FL0102-BZ_CL']-data_fl.iloc[0][2]
y3=data_fl['FL0103-BZ_CL']-data_fl.iloc[0][3]
y4=data_fl['FL0104-BZ_CL']-data_fl.iloc[0][4]
yt=-np.sqrt(y2**2+y4**2+y3**2)
yg=data_fl['Grad1']
#y2=data_fl['Grad1']
textbox=''
#%%
X_grad=np.array(x1)
Z_grad=np.array(y0)
time_ref=np.array(x2)
Z_ref= np.array(y1)
X_ref= np.array(y3)
Y_ref= np.array(y4)

origin=pd.DataFrame(data=[X_grad, Z_grad, time_ref, Z_ref, X_ref, Y_ref]).T
origin.columns=['time_grad', 'y_grad', 'time_ref', 'Z_ref', 'X_ref', 'Y_ref']
print(origin)
print(origin.shape)
origin.to_csv('heart_grad'+str(dir_name)+'_'+'whole.csv', sep=',', index=False)


#%%
fig_comparison2, ax1 = plt.subplots()

y1lim=[-0.3e-9,0.3e-9]
y2lim=[-1.5e-9,1.5e-9]

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel(mode +' [T]', color=color)
ax1.set_title('Lifts test '+dir_name)

#ax1.plot(evoked_aux.times, aux_ev_data,  color='grey', lw=0.5)
#ax1.plot(evoked_gf.times[s:f], sine_func(evoked_gf.times[s:f], params[0], params[1], params[2]), color='black')
ax1.plot(x1, y0,   color=color, zorder=10)
#ax1.plot(x2, y2, color='blue', lw=1,  zorder=0)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=y1lim)
ax1.text(-0.2, y1lim[0]*0.95, textbox)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Reference '+mode_fl+' [T]', color=color)  # we already handled the x-label with ax1
ax2.plot(x2, y1, color='pink', lw=1,  zorder=0)
ax2.plot(x2, y2, color='purple', lw=1,  zorder=0)
ax2.plot(x2, y3, color='cyan', lw=1,  zorder=0)
ax2.plot(x2, y4, color='orange', lw=1,  zorder=0)
#ax2.plot(x2, yt, color='green', lw=1,  zorder=0)
ax2.plot(x2, yg, color='gray', lw=1,  zorder=0)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=y2lim)

fig_comparison2.tight_layout()  # otherwise the right y-label is slightly clipped
fig_comparison2.savefig(str(dir_name) + '_long_run_comparison.png')
#plt.legend()
plt.show()


#%%choppedterm 4 sensors 2 orthogonal
tmin=98
d=5
tmax=tmin+d
st_ind_g=int(tmin*sfreq_g)
st_ind_fl=int(tmin*sfreq_fl)

end_ind_fl=int(tmax*sfreq_fl)
end_ind_g=int(tmax*sfreq_g)

crop_flc=data_flc[st_ind_fl:end_ind_fl]
crop_gc=data_gc[st_ind_g:end_ind_g]
print(crop_flc.shape, data_flc.shape)


x1=data_g['time']
y0=data_g['B_T_cal']-data_g.iloc[0][5]

#y1=data_g['value']
x2=data_fl['time']-data_fl.iloc[0][0]
y1=data_fl['FL0101-BZ_CL']-data_fl.iloc[0][1]
y2=data_fl['FL0102-BZ_CL']-data_fl.iloc[0][2]
y3=data_fl['FL0103-BZ_CL']-data_fl.iloc[0][3]
y4=data_fl['FL0104-BZ_CL']-data_fl.iloc[0][4]
yt=-np.sqrt(y2**2+y4**2+y3**2)
yg=data_fl['Grad1']
#y2=data_fl['Grad1']
textbox=''

fig_heart, ax1 = plt.subplots()

y1lim=[-0.02e-9,0.02e-9]
y2lim=[-0.1e-9,0.05e-9]

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel(mode +' [T]', color=color)
ax1.set_title('Real time heart measurement')

#ax1.plot(evoked_aux.times, aux_ev_data,  color='grey', lw=0.5)
#ax1.plot(evoked_gf.times[s:f], sine_func(evoked_gf.times[s:f], params[0], params[1], params[2]), color='black')
ax1.plot(x1[st_ind_g:end_ind_g], y0[st_ind_g:end_ind_g]-y0[st_ind_g], lw=0.5,   color=color, zorder=1000)
#ax1.plot(x2, y2, color='blue', lw=1,  zorder=0)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=y1lim)
ax1.text(-0.2, y1lim[0]*0.95, textbox)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'b'
ax2.set_ylabel('Reference '+mode_fl+' [T]', color=color)  # we already handled the x-label with ax1
ax2.plot(x2[st_ind_fl:end_ind_fl], y1[st_ind_fl:end_ind_fl]-y1[st_ind_fl], color='b', lw=0.5,  zorder=0)
#ax2.plot(x2, y2, color='purple', lw=1,  zorder=0)
#ax2.plot(x2, y3, color='cyan', lw=1,  zorder=0)
#ax2.plot(x2, y4, color='orange', lw=1,  zorder=0)
#ax2.plot(x2, yt, color='green', lw=1,  zorder=0)
#ax2.plot(x2, yg, color='gray', lw=1,  zorder=0)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=y2lim)

fig_comparison2.tight_layout()  # otherwise the right y-label is slightly clipped
fig_comparison2.savefig(str(dir_name) + '_long_run_comparison.png')
#plt.legend()
fig_heart.set_zorder(ax2.get_zorder()+10)
#fig_heart.set_frame_on(False)
plt.show()

#%%##export data for origin

x_grad=np.array(x1[st_ind_g:end_ind_g])
y_grad=np.array(y0[st_ind_g:end_ind_g])
x_ref=np.array(x2[st_ind_fl:end_ind_fl])
y_ref= np.array(y1[st_ind_fl:end_ind_fl])
origin=pd.DataFrame(data=[x_grad, y_grad, x_ref, y_ref]).T
origin.columns=['x_grad', 'y_grad', 'x_ref', 'y_ref']
print(origin)
print(origin.shape)
origin.to_csv('heart_grad'+str(dir_name)+'_'+str(tmin)+'s.csv', sep=',', index=False)
