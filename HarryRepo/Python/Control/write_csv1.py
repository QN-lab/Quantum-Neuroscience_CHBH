# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:18:53 2023

@author: vpixx
"""
import numpy as np
import matplotlib.pyplot as plt
import os

length = np.ones(5000) #

A = np.array([0,1.25,1.25,1.5,1.5,1.75,1.75,0])

conv = 3.23e-3 #Vamp/field

sig = np.zeros((len(A),len(length)))

for i in range(len(A)):
    sig[i,:] = A[i]*conv*length
    
output = sig.flatten()

plt.figure()
plt.plot(output)
plt.title('input DC Values')
plt.xlabel('samples(10kSa/s)')
plt.ylabel('mV input into Sig Gen')

os.chdir('Z:\\Data\\2023_08_17_bench\\')

np.savetxt("DC_mag_step_positive_17-08-23_1.25-1.75nT.csv", output, delimiter=",")