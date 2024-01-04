# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:53:11 2023

@author: kowalcau
"""
############# run only once this part ( after running mne part)
st_ind=int(events_m_fl[0,0]+tmin_e*sfreq_fl)
end_ind=int(events_m_fl[-1,0]+tmax_e*sfreq_fl)
end_ind_g=int(events_g[-1,0]+tmax_e*sfreq_g)

data_fl=data_fl[st_ind:end_ind]
data_fl.shape
#%% crop both to the same length
tcrop=540
data_g=data_g[0:int(tcrop*sfreq_g)]
data_fl=data_fl[0:int(tcrop*sfreq_fl)]
#%% long term plot
x1=data_g['time']
y1=data_g['B_T_cal']-data_g.iloc[0][5]
print(data_g.head(1), data_fl.head(1))


#y1=data_g['value']
x2=data_fl['time']-data_fl.iloc[0][0]
y2=data_fl['FL0101-BZ_CL']-data_fl.iloc[0][1]
#y3=data_fl['FL0102-BZ_CL']-data_fl.iloc[0][2]
#y4=data_fl['FL0103-BZ_CL']-data_fl.iloc[0][3]
#y5=data_fl['FL0104-BZ_CL']-data_fl.iloc[0][4]
#y6=-np.sqrt(y2**2+y4**2+y3**2)
#data_fl['Grad1']=data_fl['FL0101-BZ_CL']-data_fl['FL0102-BZ_CL']
#y2=data_fl['Grad1']
textbox=''
fig_comparison2, ax1 = plt.subplots()
y2lim=[-0.5e-9,0.25e-9]
y1lim=[-1e-9,1e-9]

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel(mode +' [T]', color=color)
ax1.set_title('Heart measurement')#+dir_name)

#ax1.plot(evoked_aux.times, aux_ev_data,  color='grey', lw=0.5)
#ax1.plot(evoked_gf.times[s:f], sine_func(evoked_gf.times[s:f], params[0], params[1], params[2]), color='black')
ax1.plot(x1, y1,   color=color, zorder=10)
#ax1.plot(x2, y2, color='blue', lw=1,  zorder=0)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=y2lim)
ax1.text(-0.2, y2lim[0]*0.95, textbox)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Reference '+mode_fl+' [T]', color=color)  # we already handled the x-label with ax1
ax2.plot(x2, y2, color=color, lw=1,  zorder=0)
# ax2.plot(x2, y3, color='purple', lw=1,  zorder=0)
# ax2.plot(x2, y4, color='cyan', lw=1,  zorder=0)
# ax2.plot(x2, y5, color='orange', lw=1,  zorder=0)
# ax2.plot(x2, y6, color='green', lw=1,  zorder=0)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=y2lim)

fig_comparison2.tight_layout()  # otherwise the right y-label is slightly clipped
fig_comparison2.savefig(str(dir_name) + '_long_run_comparison.png')
#plt.legend()
plt.show()
#%% Selection for the heartbeat data
x1=data_g['time']
y1=(data_g['B_T_cal']-data_g.iloc[0][5])*1
#y3=data_g['error_deg']*180/np.pi#-data_g.iloc[0][5]
#y1=data_g['value']
x2=data_fl['time']-data_fl.iloc[0][0]
y2=data_fl['FL0101-BZ_CL']-data_fl.iloc[0][1]
data_fl.head(1)
#data_fl['Grad1']=data_fl['FL0101-BZ_CL']-data_fl['FL0102-BZ_CL']
#y2=data_fl['Grad1']


start=260
stop=start+20
fig_heart, ax1 = plt.subplots()
y1lim=[0.5e-10,1.5e-10]
y2lim=[-1e-10,2e-10]
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
fig_heart.savefig(str(dir_name) + '_start_'+str(start) +'_heart.png')
#plt.legend()
plt.show()


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