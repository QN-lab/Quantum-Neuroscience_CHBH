# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:03:25 2023

@author: hxc214
"""

import numpy as np
import math
import matplotlib.pyplot as plt


w = 50
f0 = 1500

#%% Change only fe, fix fg =0

fe = np.linspace(-50,50,160)
fg = 0

F1 = np.zeros(160)
F2 = np.zeros(160)

for i in range(len(fe)):
    if w>fg+2*fe[i]:
        F1[i] = f0+fg/2
        F2[i] = f0+fg/2
    else:
        F1[i] = f0 + fg/2 + (np.sqrt(4*fe[i]**2+4*fe[i]*fg+fg**2-w**2))/2
        F2[i] = f0 + fg/2 - (np.sqrt(4*fe[i]**2+4*fe[i]*fg+fg**2-w**2))/2

plt.figure()
plt.plot(fe,F1)
plt.plot(fe,F2)

#%% Change both fg and fe

fe = np.linspace(-50,50,160)
# fg = np.linspace(-0.35,0.35,160)
fg = np.linspace(-50,50,160)

ee, gg = np.meshgrid(fe,fg)

# F = f0 + fg/2 + (np.sqrt(4*fe**2+4*fe*fg+fg**2-w**2))/2

F1 = np.zeros((160,160))
F2 = np.zeros((160,160))

for i in range(len(fe)):
    for j in range(len(fg)):
        if w>gg[i,j]+2*ee[i,j]:
            F1[i,j] = f0+gg[i,j]/2
            F2[i,j] = f0+gg[i,j]/2
        elif ee[i,j]>= 0:
            F1[i,j] = f0 + gg[i,j]/2 + (np.sqrt(4*ee[i,j]**2+4*ee[i,j]*gg[i,j]+gg[i,j]**2-w**2))/2
            F2[i,j] = f0 + gg[i,j]/2 - (np.sqrt(4*ee[i,j]**2+4*ee[i,j]*gg[i,j]+gg[i,j]**2-w**2))/2
        elif ee[i,j]<= 0:
            F1[i,j] = f0 + gg[i,j]/2 + (np.sqrt(4*ee[i,j]**2+4*ee[i,j]*gg[i,j]+gg[i,j]**2-w**2))/2
            F2[i,j] = f0 + gg[i,j]/2 - (np.sqrt(4*ee[i,j]**2+4*ee[i,j]*gg[i,j]+gg[i,j]**2-w**2))/2

ax=plt.figure().add_subplot(projection='3d')

ax.plot_surface(ee,gg,F1,edgecolor='royalblue', lw=0.5, rstride=20, cstride=20,
                alpha=0.3)