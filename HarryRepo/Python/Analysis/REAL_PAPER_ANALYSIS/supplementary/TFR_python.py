# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:23:03 2023

@author: hxc214
"""


# import obs
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import regex as re
import pandas as pd
import os
import math
# import Harry_analysis as HCA
from scipy.fft import fft, fftfreq

#%%

# IPython_default = plt.rcParams.copy()

# plt.rc('_internal.classic_mode':False)

#%%

os.chdir('Z:\\jenseno-opm\\Publications_submission\\Gradiometer Paper Prep\\Origin files\\')
# os.chdir('Z:\\jenseno-opm\\Publications_submission\\Gradiometer Paper Prep\\Origin files\\')
dat = pd.read_csv('itc_export_g_3.csv')
# dat = pd.read_csv('trf_export_fl.csv')

mapped = dat.pivot_table(index='time',columns='freq',values='B_T_cal')
# mapped = dat.pivot_table(index='time',columns='freq',values='FL0101-BZ_CL')
# mapped = mapped.iloc[:,2:]

mapped_arr= mapped.to_numpy().transpose()

with plt.style.context('default'):
    colors = 'Reds'
    
    tfr = plt.imshow(mapped_arr,
              aspect = 'auto',
              origin='lower',
              extent=[mapped.index[0],mapped.index[-1],mapped.keys()[0],mapped.keys()[-1]],
              cmap=mpl.colormaps[colors],
              interpolation =  'none'
              )
    
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency(Hz)')
    plt.colorbar(tfr,label='Inter-trial Coherence (arb)')
    
    # plt.clim(np.min(mapped_arr),np.max(mapped_arr)*1)
    # ax.set_xticks(ticks = np.arange(mapped_arr.shape[1]),labels= list(mapped.columns))
    plt.show()
