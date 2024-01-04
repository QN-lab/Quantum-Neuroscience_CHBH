# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:25:58 2023

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')

yerr = 2

c_f = np.array([1.515295,1.515295,1.535643,1.535643,1.554147,1.554147,
                1.590656,1.590656,1.609955,1.609955,1.629008,1.629008,1.5696,1.5696])

p_f = np.array([32,68,13,88,6,95,42,59,62,36,18,82,19,80])

plt.errorbar(c_f,p_f,yerr = yerr, fmt='b.')
plt.xlabel('Central lock frequency (kHz)')
plt.ylabel('Location of noise on PID spectrum (Hz)')
plt.grid()