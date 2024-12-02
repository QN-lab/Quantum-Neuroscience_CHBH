from Proc import obs
# import obs

import matplotlib.pyplot as plt

import numpy as np
import regex as re
import pandas as pd
import os
import math
import Harry_analysis as HCA
from scipy.fft import fft, fftfreq
plt.rcParams['text.usetex'] = True
plt.style.use('default')

#%% Directories and load data

r_sig = 'dev3994_demods_0_sample_00000.csv'
r_header = 'dev3994_demods_0_sample_header_00000.csv'

nice_res_dir = 'Z:\\Data\\2023_08_09_bench\\most_perfect_resonance_000\\'

res_g, res_legends_g = HCA.ReadData(nice_res_dir,r_sig,r_header,';')
resonance_g = obs.Resonance(res_g, res_legends_g, 1100)

nice_res_f = resonance_g.data[0,:,0]
nice_res_x = resonance_g.data[0,:,1]
nice_res_y = resonance_g.data[0,:,2]
nice_res_phi = resonance_g.data[0,:,3]

fig,ax= plt.subplots()

ax.plot(nice_res_f,nice_res_x/1e-3,c='green')
ax.plot(nice_res_f,nice_res_y/1e-3,c='blue')
ax.axhline(y=0,c='k',ls='--')
ax.grid()

ax.set(xlabel='Modulation Frequency (Hz)',ylabel='Rotation (arb,mV)',ylim=[-4,8],xlim=[1000,1900])



# ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axi
# ax2.plot(nice_res_f,nice_res_phi,c='red')

