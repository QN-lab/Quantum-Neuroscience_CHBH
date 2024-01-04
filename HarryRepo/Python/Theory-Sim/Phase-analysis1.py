# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:17:39 2023

@author: hxc214
"""


import numpy as np
import matplotlib.pyplot as plt

w = 50

fe = np.linspace(-20,20,160)
fg = np.linspace(-0.35,0.35,160)

fem,fgm = np.meshgrid(fe,fg)

feplot = fem/14
fgplot = fgm/14

F = 1*np.arctan((1/w)*((-(fem)**2-fem*fgm+0.25*w**2)*fgm)/
                 (fem**2+fem*fgm+0.5*fgm**2+0.25*w**2))

fg0 = np.zeros(fe.shape[0])

Fr = 1*np.arctan((w*fg)/(w**2+2*fg))

ax=plt.figure().add_subplot(projection='3d')

ax.plot_surface(feplot,fgplot,F/1e-3,edgecolor='royalblue', lw=0.5, rstride=20, cstride=20,
                alpha=0.3)
ax.set_xlabel('Applied B field (nT)')
ax.set_ylabel('Applied G field (pT/cm) ')
ax.set_zlabel('Phase (arb)')
ax.set_title('Surface plot for width = '+str(w))

ax.plot(fe/7,fg0,fg0,linewidth=5,label='fg=0')
ax.plot(fg0,fg/7,Fr/1e-3,linewidth=5,label='fe=0')
ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.9))


#%% 
slope = np.zeros(len(fe))

for i in range(len(fe)):
   s = F[i,:]
   m,b = np.polyfit(fg,s,1)
   slope[i] = m

plt.figure()
plt.plot(fe/7,slope/1e-4)
plt.xlabel('applied B field(nT)')
plt.ylabel('slope of the phase (Arb)')
plt.title('THIS DOESNT MAKE SENSE')
