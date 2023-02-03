# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:48:05 2023

@author: vpixx
"""
import numpy as np
import matplotlib.pyplot as plt
import Harry_analysis as HCA


B = np.array([0,1,5,10,20,50]) #nT
B_corr = B
central_f = np.array([1.6983, 1.726610311719,1.839929086124,1.982317711476,
                          2.266919623597,3.117296156455])*1e3

del_f = (central_f - central_f[0])

del_B = del_f*0.071488 #nT/Hz

m,b = np.polyfit(B_corr, del_B, 1)

plt.figure()
plt.plot(B_corr,del_B,'k*')
plt.axline(xy1=(0, b), slope=m,label= 'slope: {} gain, intercept: {} nT'.format(m,b))
plt.title('Measured field due to input field from corrected MS-2 Calibration')
plt.xlabel('Applied field from HH coil (nT)')
plt.ylabel('Measured field due to resonance shift (nT)')
plt.xlim(-5,60)
plt.legend()
plt.ylim(-10,220)
