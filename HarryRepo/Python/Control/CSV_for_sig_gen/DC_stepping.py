# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:39:45 2023

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt

length = np.ones(10000)

A = [0,0.5,1,2,3,5,7.5,10]

pp_conv = 3.23e-3 # V(amp)/nT

sig = np.zeros((len(A),len(length)))

for i in range(len(A)):
    sig[i,:] = A[i]*pp_conv*length
    
output = sig.flatten()

plt.figure()
plt.plot(output)

# np.savetxt("DC_HG_Steps_0-10nT.csv", output, delimiter=",")