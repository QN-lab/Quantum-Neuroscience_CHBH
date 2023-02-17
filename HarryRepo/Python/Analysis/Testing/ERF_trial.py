# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:26:53 2023

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
import math
import Harry_analysis as HCA
import regex as re
from scipy.fft import fft, fftfreq
from scipy import signal
import statistics as stats

csv_sep = ';'

daq_0 = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230215_232845_01/grad_ERR_000/'

trace_0, trace_legends_0 = HCA.DAQ_read_shift(daq_0,csv_sep) 

track0 = HCA.DAQ_Tracking_PURE(trace_0,trace_legends_0)

averaged = np.mean(track0.chunked_data,axis=1)

plt.figure()
plt.plot(track0.chunked_time[0,:],averaged)
plt.xlabel('Time (s)')
plt.ylabel('Field (Hz)')
plt.title('Measured signal averaged over {} runs'.format(len(track0.patch)))

