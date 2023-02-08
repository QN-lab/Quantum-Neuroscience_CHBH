# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:57:22 2023

@author: H

Writing the code setup for comparing parameters
"""

#TO Do
#Error on both parameters

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# plt.style.use('classic')
import math
import Harry_analysis as HCA

#%% inputs
csv_sep = ';'
Param = [10, 25, 50, 100, 200]
P = 25 #P value

looper = range(len(Param))

T_trigger = list([])
aux = list([])
track = list([])

for i in looper:
    daq = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230206_144958_01/slew_{}BW_P-{}_000/'.format(Param[i],P)
    
    trig, trig_legends = HCA.DAQ_trigger_read(daq, csv_sep)
    aux_i, aux_legends = HCA.DAQ_read_auxin0(daq, csv_sep)
    track_i, track_legends = HCA.DAQ_tracking_read(daq, csv_sep)
    
    T_trigger.append(HCA.DAQ_Trigger(trig, trig_legends))
    aux.append(HCA.DAQ_Tracking(aux_i, aux_legends,T_trigger[i]))
    track.append(HCA.DAQ_Tracking(track_i,track_legends,T_trigger[i]))
    
#%% Average over cleaned, chunked runs. Within each object

for i in looper:
    aux[i].averaged = np.mean(aux[i].cleaned_chunked,axis=0)
    track[i].averaged = np.mean(track[i].cleaned_chunked,axis=0)
    
plt.figure()
plt.plot(aux[1].averaged)
plt.plot(track[1].averaged)
    