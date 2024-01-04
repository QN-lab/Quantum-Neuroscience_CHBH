# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:25:58 2023

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')

yerr = 2
c_f = np.array([1.510918,1.465982,1.848353,2.214589,2.214589,2.603450,2.603450,2.996679,3.384472,5.319564])

p_f = np.array([+41.5,-13,+5,+32.2,+36.0,+44,-56.4,-51.5,-36.5,35.4])

p = np.array([3.54,3.42,3.42,0.877,1.00,3.17,0.945,0.981,1.00,1.50])

plt.errorbar(c_f,p_f,yerr = yerr, fmt = 'b*')
plt.xlabel('Central lock frequency (kHz)')
plt.ylabel('Location of noise on Power spectrum (Hz)')
plt.grid()