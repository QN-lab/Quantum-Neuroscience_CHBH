# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:57:56 2024

@author: vpixx
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


pvals = np.array([99.00759,
    99.87268,
    100.67707,
    101.39545,
    102.09865])

df = pd.read_csv('Z:\\Data\\TEST_hb.csv').to_numpy()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

indi = np.zeros(len(pvals))

for i in range(len(pvals)):
    indi[i] = find_nearest(df[:,0],pvals[i])


indii = indi-200
indf = indi+500

mat = np.zeros((len(pvals),700))

for i in range(len(pvals)):
    mat[i,:] = df[int(indii[i]):int(indf[i]),1]
    
    
avg = np.mean(mat,axis=0)


plt.plot(avg)