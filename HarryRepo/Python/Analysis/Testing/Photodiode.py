# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:19:16 2023

@author: vpixx
"""
#MODULATION ON FOR THESE RUNS

import numpy as np
import matplotlib.pyplot as plt
Pow = np.array([6.8,17.6,27.7,34.2,40.4,46.1,51.5]) # uW
Pow_err = 0.5

PD = np.array([8.3,25.9,40.6,53.6,63.1,69.5,77.8])
PD_err = 0.3

plt.figure()
plt.errorbar(Pow,PD,yerr = Pow_err, xerr = PD_err,fmt = 'k*')
plt.xlabel('Laser Power (uW)')
plt.ylabel('Photodiode response (mV)')
plt.grid()

#MODULATION ON FOR THESE RUNS


m,b = np.polyfit(Pow,PD, 1)

plt.axline(xy1=(0, b), slope=m,label= 'slope: {} nT/mA, intercept: {} nT'.format(m,b))

print('m = '+ str(m) + ', b = ' + str(b))