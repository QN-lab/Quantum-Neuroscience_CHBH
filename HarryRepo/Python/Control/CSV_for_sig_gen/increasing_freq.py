# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:39:45 2023

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt

length = np.linspace(0, 1,num=10000)

f = [8,16,24,32,40,48,56,80]

A = 3.23e-3

sig = np.zeros((len(f),len(length)))

for i in range(len(f)):
    sig[i,:] = A*np.sin(2*np.pi*f[i]*length)
    
output = sig.flatten()

plt.figure()
plt.plot(output)

np.savetxt("atten_freq.csv", output, delimiter=",")