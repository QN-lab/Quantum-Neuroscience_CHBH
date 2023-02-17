# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:43:38 2023

@author: vpixx
"""
import numpy as np
import matplotlib.pyplot as plt#

length = np.ones(5000)

pt_0 = 0*length

pt_05 = 1.54e-3*length

pt_1 = 3.07e-3*length

pt_15 = 4.61e-3*length

pt_2 = 6.15e-3*length

pt_5 = 1.54e-2*length

pt_10 = 3.07e-2*length

pt_20 = 6.15e-2*length

go_back = np.linspace(pt_20[0],pt_0[0],num=5000)

output = np.hstack([pt_0,pt_05,pt_1,pt_15,pt_2,pt_5,pt_10,pt_20,go_back])

plt.figure()
plt.plot(output)


np.savetxt("0_20pT.csv", output, delimiter=",")