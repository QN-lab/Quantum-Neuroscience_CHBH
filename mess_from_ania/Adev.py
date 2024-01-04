# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:17:48 2023

@author: kowalcau
"""

#Import libraries
import numpy as np
import math
import allantools as at
import pandas as pd


t = np.logspace(-2, 2, 50)
y = data_g.iloc[0:1000,5]
r = sfreq_g

(t2, ad, ade, adn) = at.oadev(y, rate=r, data_type="freq", taus=t)  # Compute the overlapping ADEV
fig = plt.loglog(t2, ad) # Plot the results
plt.show()

b = at.Plot()
b.plot(a, errorbars=True, grid=True)
b.ax.set_xlabel("Tau (s)")
b.show()
#tauArray=[0.01,0.1,0.5, 1,5, 10]
tau_g, Adev_g 
tau_m, Adev_m = cal_oadev(data_m['B_T_cal'],sfreq_g,tauArray)


fig_fft_adev, ax1 = plt.subplots()
ylim=[1e-16,10e-10]
color = 'tab:red'
ax1.set_xlabel('Averaging time (s)')
ax1.set_ylabel('Allan deviation (T)', color=color)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.fill
ax1.set_title('Allan deviaiton ' +str(dir_name))

ax1.plot(tau_g, Adev_g, color='darkred')
#ax1.fill_between(freqs_gf[freq_range_g], psds_gf_mean - psds_gf_std, psds_gf_mean + psds_gf_std,     color=color, alpha=.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=ylim)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Magnetometer Allan Deviation (T)', color=color)  # we already handled the x-label with ax1
ax2.set_yscale('log')

ax2.plot(tau_m, Adev_m, color='darkblue')
#ax2.fill_between(freqs_mf[freq_range_m], psds_m_mean - psds_m_std, psds_m_mean + psds_m_std,     color=color, alpha=.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=ylim)
ax2.set_xscale('log')
fig_fft_adev.tight_layout()  # otherwise the right y-label is slightly clipped
fig_fft_adev.savefig(str(dir_name) + '_adev.png')
plt.show()