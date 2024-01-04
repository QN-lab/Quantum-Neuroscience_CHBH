# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:01:39 2023

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
import pandas as pd

Vin = np.array([0.15,0.3,0.6,1,2,3,4,6,8,9,10])/2
Vout = np.array([0.22,0.54,1.1,1.85,3.76,5.50,7.41,11.21,14.83,16.71,18.7])
Vy_err = 0.1

m, b = np.polyfit(Vin, Vout, 1)

plt.figure()
plt.errorbar(Vin,Vout,yerr=Vy_err,fmt = 'o')
plt.title('Voltage Divider response')
plt.xlabel('Voltage input (Amplitude), V')
plt.ylabel('Measured signal after divider (mV)')
plt.grid()
plt.text(0,15,'m = {}; b = {}'.format(m,b))

