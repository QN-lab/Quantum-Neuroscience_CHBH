# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:06:32 2023

@author: H
"""
# import numpy as np
import matplotlib.pyplot as plt

#plotting Atten vs amplitude

Field = [8,16,24,32,40,48,56,80]

nT_0 = [28.7636,27.0342,25.8378,25.1882,25.2031,25.5353,26.1089,26.9296]

nT_60 = [37.0227,32.8226,30.4089,28.4238,27.094,25.9978,25.3577,24.4576]

plt.figure()
plt.plot(Field,nT_0,'-o',label = '0nT')
plt.plot(Field,nT_60,'-o',label = '60nT')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation (dB)')
plt.title('Grad vs Mag attenuation of Frequency')
plt.grid(color='k', linestyle='-', linewidth=0.3)
# plt.xlim(-0.5,6)
# plt.ylim(20, 35)
plt.legend(title = 'Lock level',loc = 'upper right')