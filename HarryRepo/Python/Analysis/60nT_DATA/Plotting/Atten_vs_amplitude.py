# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:06:32 2023

@author: H
"""

# import numpy as np
import matplotlib.pyplot as plt

#plotting Atten vs amplitude

Field = [0.5,0.75,1,2,3,5]

Hz8_0nT = [39.6048,39.293,42.4971,41.8918,50.6599,28.3471]

Hz8_60nT = [35.0323,35.4889,38.3713,50.0327,48.8205,38.0338]

Hz80_0nT = [43.1816,42.7936,44.8342,44.1223,43.8647,43.6834]

Hz80_60nT = [41.4075,41.6909,42.461,43.3122,43.9302,43.62]

plt.figure()
plt.plot(Field,Hz8_0nT,'-o',label = '8Hz at 0nT')
plt.plot(Field,Hz8_60nT,'-o',label = '8Hz at 60nT')
plt.plot(Field,Hz80_0nT,'-o',label = '80Hz at 0nT')
plt.plot(Field,Hz80_60nT,'-o',label = '80Hz at 60nT')
plt.xlabel('Applied Field (nT)')
plt.ylabel('Attenuation (dB)')
plt.title('Grad vs Mag atteunuation of 0.5nT HG sine')
plt.grid(color='k', linestyle='-', linewidth=0.3)
plt.xlim(-0.5,6)
# plt.ylim(20, 35)
plt.legend(title = 'Frequency and lock level',loc = 'upper right')