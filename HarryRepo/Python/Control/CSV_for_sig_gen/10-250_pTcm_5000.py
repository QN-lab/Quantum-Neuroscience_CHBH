# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:46:42 2023

@author: vpixx
"""

import numpy as np



length = np.ones(5000)

pt_10 = 1.537e-2*length

pt_25 = 3.841e-2*length

pt_50 = 7.683e-2*length

pt_75 = 1.152e-1*length

pt_100 = 1.537e-1*length

pt_150 = 2.305e-1*length

pt_200 = 3.073e-1*length

pt_250 = 3.841e-1*length

go_back = np.linspace(pt_250[0],pt_10[0],num=5000)

output = np.hstack([pt_10,pt_25,pt_50,pt_75,pt_100,pt_150,pt_200,pt_250,go_back])
