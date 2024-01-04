# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:39:45 2023

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt

t = [0.5,0.6,0.4,0.6,0.4,0.6,0.4,0.6,0.5]

A = np.array([0,8.22,0,0.124,0,0.412,0,16.48,0])*0.5

length = np.linspace(0,1,int((sum(t)*10000)))

f = 1 #Hz

sig = []

for i in range(len(A)):
    sig.append((A[i]*np.sin(2*np.pi*f*np.linspace(0,1,int(10000*t[i]))))+0.0823)
    
output = np.hstack(sig)

plt.figure()
plt.plot(output)

np.savetxt("Brain_Sim_4_22-2-23.csv", output, delimiter=",")