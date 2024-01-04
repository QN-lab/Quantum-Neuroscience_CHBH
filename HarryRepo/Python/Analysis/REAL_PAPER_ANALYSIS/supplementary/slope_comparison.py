# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:03:10 2023
 MAGNETOMETER VS GRADIOMETER SIGNALS
@author: hxc214
"""

import numpy as np
import sys
#import sounddevice as sd
#import serial
#from simple_pid import PID
#import winsound
import regex as re
# from Proc import obs
import obs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

#%% Data Extraction
filename = 'dev3994_demods_0_sample_00000.csv'
headername = 'dev3994_demods_0_sample_header_00000.csv'

g_loc = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\grad_noise_res\\grad_noise_res_000\\'
m_loc = 'Z:\\jenseno-opm\\Data\\2023_08_17_bench\\mag_noise_res\\mag_noise_res_000\\'

g_sig, g_legends = obs.PhaseReadData(g_loc,filename,headername,';')
m_sig, m_legends = obs.PhaseReadData(m_loc,filename,headername,';')

g_res = obs.Resonance(g_sig,g_legends,1100)
m_res = obs.Resonance(m_sig,g_legends,1100)

fg_in = g_sig.index[g_sig['fieldname'] == 'frequency'].tolist()
pg_in = g_sig.index[g_sig['fieldname'] == 'phase'].tolist()
xg_in = g_sig.index[g_sig['fieldname'] == 'x'].tolist()
yg_in = g_sig.index[g_sig['fieldname'] == 'y'].tolist()

fm_in = m_sig.index[m_sig['fieldname'] == 'frequency'].tolist()
pm_in = m_sig.index[m_sig['fieldname'] == 'phase'].tolist()
xm_in = m_sig.index[m_sig['fieldname'] == 'x'].tolist()
ym_in = m_sig.index[m_sig['fieldname'] == 'y'].tolist()

fg_data = g_sig.iloc[fg_in[1],4:-1].to_numpy()
pg_data = g_sig.iloc[pg_in[1],4:-1].to_numpy()
xg_data = g_sig.iloc[xg_in[1],4:-1].to_numpy()
yg_data = g_sig.iloc[yg_in[1],4:-1].to_numpy()


g_dat = np.zeros((4,len(fg_data)))

g_dat[0,:] = fg_data
g_dat[1,:] = xg_data
g_dat[2,:] = yg_data
g_dat[3,:] = pg_data

fm_data = m_sig.iloc[fm_in[1],4:-1].to_numpy()
pm_data = m_sig.iloc[pm_in[1],4:-1].to_numpy()
xm_data = m_sig.iloc[xm_in[1],4:-1].to_numpy()
ym_data = m_sig.iloc[ym_in[1],4:-1].to_numpy()

m_dat = np.zeros((4,len(fm_data)))

m_dat[0,:] = fm_data
m_dat[1,:] = xm_data
m_dat[2,:] = ym_data
m_dat[3,:] = pm_data

#%% Plot data
fig,ax = plt.subplots(3,1,layout='constrained')
# ax.ticklabel_format(useOffset=False, style='plain')

ax[0].plot(m_dat[0,:],m_dat[1,:]/1e-3,'b', label='mag')
ax[0].plot(g_dat[0,:],g_dat[1,:]/1e-3,'g', label= 'grad')
ax[0].set_ylabel('quad, mV')
ax[0].legend()

ax[1].plot(m_dat[0,:],m_dat[2,:]/1e-3,'b')
ax[1].plot(g_dat[0,:],g_dat[2,:]/1e-3,'g')
ax[1].set_ylabel('in-phase,mV')

ax[2].plot(m_dat[0,:],m_dat[3,:],'b')
ax[2].plot(g_dat[0,:],g_dat[3,:],'g')
ax[2].set_ylabel('Phase')
ax[2].set_xlabel('frequency (Hz)')


def find_inx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

g_roi = [g_res.central_f[0]-g_res.width[0]/4, g_res.central_f[0]+g_res.width[0]/4]
g_inx = [find_inx(g_dat[0,:],g_roi[0]), find_inx(g_dat[0,:],g_roi[1])]

m_roi = [m_res.central_f[0]-m_res.width[0]/4, m_res.central_f[0]+m_res.width[0]/4]
m_inx = [find_inx(m_dat[0,:],m_roi[0]), find_inx(m_dat[0,:],m_roi[1])]

for i in g_roi:
    ax[2].axvline(x=i,c='g',ls='--')
    
for i in m_roi:
    ax[2].axvline(x=i,c='b',ls='--')

#%% Phase analysis!!

g_dat_r = (g_dat[:,g_inx[0]:g_inx[1]])
m_dat_r = (m_dat[:,m_inx[0]:m_inx[1]])

g_coef = np.polyfit(g_dat_r[0,:],g_dat_r[3,:],1)
g_poly = np.poly1d(g_coef) 

m_coef = np.polyfit(m_dat_r[0,:],m_dat_r[3,:],1)
m_poly = np.poly1d(m_coef) 


fig2,ax2 = plt.subplots()
ax2.ticklabel_format(useOffset=False)

ax2.plot(g_dat_r[0,:],g_dat_r[3,:],'go',g_dat_r[0,:],g_poly(g_dat_r[0,:]),'g--', label = 'grad')
ax2.plot(m_dat_r[0,:],m_dat_r[3,:],'bo',m_dat_r[0,:],m_poly(m_dat_r[0,:]),'b--', label = 'mag')
ax2.set_ylabel('Phase')
ax2.set_xlabel('Frequency(Hz)')

print('Grad slope: '+str(g_coef[0].round(3)))
print('Mag slope: '+str(m_coef[0].round(3)))
print()


sr = g_coef[0]/m_coef[0]

iwr = m_res.width[0]/g_res.width[0]

print('Slope ratio: ' +str(sr.round(3)))
print()
print('Grad width: '+str(g_res.width[0].round(3)))
print('Mag width: '+str(m_res.width[0].round(3)))
print()
print('Inverse Width ratio: '+ str(iwr.round(2)))
print()


#%% In-Phase analysis

g_dat_r = (g_dat[:,g_inx[0]:g_inx[1]])
m_dat_r = (m_dat[:,m_inx[0]:m_inx[1]])

g_coef = np.polyfit(g_dat_r[0,:],g_dat_r[2,:],1)
g_poly = np.poly1d(g_coef) 

m_coef = np.polyfit(m_dat_r[0,:],m_dat_r[2,:],1)
m_poly = np.poly1d(m_coef) 


fig2,ax2 = plt.subplots()
ax2.ticklabel_format(useOffset=False)

ax2.plot(g_dat_r[0,:],g_dat_r[2,:],'go',g_dat_r[0,:],g_poly(g_dat_r[0,:]),'g--', label = 'grad')
ax2.plot(m_dat_r[0,:],m_dat_r[2,:],'bo',m_dat_r[0,:],m_poly(m_dat_r[0,:]),'b--', label = 'mag')
ax2.set_ylabel('In-Phase, (V)')
ax2.set_xlabel('Frequency (Hz)')

print('Grad slope: '+str(g_coef[0].round(5)))
print('Mag slope: '+str(m_coef[0].round(5)))
print()


sr = g_coef[0]/m_coef[0]

iwr = m_res.width[0]/g_res.width[0]

print('Slope ratio: ' +str(sr.round(3)))
print()
print('Grad width: '+str(g_res.width[0].round(3)))
print('Mag width: '+str(m_res.width[0].round(3)))
print()
print('Inverse Width ratio: '+ str(iwr.round(2)))
print()
