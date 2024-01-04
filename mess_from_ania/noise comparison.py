# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:18:12 2023

@author: kowalcau
"""

#print(psdsgf_mean.shape, psdsfl_mean.shape)

# fft_c1=np.sqrt(psds_gf)
# psds_c1_mean = fft_c1.mean(axis=axis)[freq_range_g]
# psds_c1_std = fft_c1.std(axis=(0, 1))[freq_range_g]


fft_flf=np.sqrt(psds_flf)
psds_fl_mean = fft_flf.mean(axis=(0, 1))[freq_range_fl]
psds_fl_std = fft_flf.std(axis=(0, 1))[freq_range_fl]

#print(psdsgf_mean.shape, psdsfl_mean.shape)

#%%
fig_fft_comparison, ax1 = plt.subplots()
ylim=[1e-15,10e-12]
color = 'tab:red'
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('FFT Gradiometer fT/sqrt(Hz)', color=color)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.fill
ax1.set_title('FFT ' +str(dir_name))
ax1.plot(freqs_gf[freq_range_g], psds_gf_mean[freq_range_g], color=color)
ax1.plot(freqs_gf[freq_range_g], psds_c1_mean[freq_range_g], color='green')
#ax1.fill_between(freqs_gf[freq_range_g], psds_gf_mean - psds_gf_std, psds_gf_mean + psds_gf_std,     color=color, alpha=.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set(ylim=ylim)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('FFT Fieldline fT/sqrt(Hz)', color=color)  # we already handled the x-label with ax1
ax2.set_yscale('log')
ax2.plot(freqs_flf[freq_range_fl], psds_fl_mean, color=color)
#ax2.fill_between(freqs_flf[freq_range_fl], psds_fl_mean - psds_fl_std, psds_fl_mean + psds_fl_std,     color=color, alpha=.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set(ylim=ylim)

fig_fft_comparison.tight_layout()  # otherwise the right y-label is slightly clipped
fig_fft_comparison.savefig(str(dir_name) + '_all_fft.png')
plt.show()
