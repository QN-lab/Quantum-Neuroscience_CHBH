# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:39:45 2023

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt

length = np.linspace(0, 1,num=10000)

f = 1

A = np.array([0,750,0,150,0,30,0,300,0])*1e-3

sig = np.zeros((len(A),len(length)))

for i in range(len(A)):
    sig[i,:] = A[i]*np.sin(2*np.pi*f*length)
    
output = sig.flatten()

plt.figure()
plt.plot(output)

np.savetxt("Brain_Sim2.csv", output, delimiter=",")