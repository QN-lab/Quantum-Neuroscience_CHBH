# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:39:45 2023

@author: vpixx
"""

import numpy as np
import matplotlib.pyplot as plt

length = np.ones(3000) #half-second

inc = 0.3

A = np.array([0,1,0,2,0,3,0,4,0,5,0,6,0,7,0,-1,0,-2,0,-3,0,-4,0,-5,0,-6,0,-7,0])*inc

pp_conv = 3.23e-3 # V(amp)/pT/cm

qq = np.array(A)*pp_conv*1e3

sig = np.zeros((len(A),len(length)))

for i in range(len(A)):
    sig[i,:] = A[i]*pp_conv*length
    
output = sig.flatten()

plt.figure()
plt.plot(output)
plt.title('input DC Values')
plt.xlabel('samples(3kSa/s)')
plt.ylabel('mV input into Sig Gen')

# diff_A.sort()

# plt.figure()
# plt.plot(diff_A,'o')
# plt.title('sorted changes in field, in ascending order against voltage difference')
# plt.xlabel('number of changes in DC field')
# plt.ylabel('difference in field (pT/cm)')

np.savetxt("DC_grad_steps_2_27-02-23.csv", output, delimiter=",")
