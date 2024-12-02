# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:52:24 2024

@author: kowalcau
"""

import mne
import numpy as np
import scipy
from scipy.spatial import ckdtree
import pathlib
import matplotlib
import matplotlib.pyplot as plt
import PyQt5
import ipympl
import IPython
# import PySide2
import os

!pwd
%run PreprocessingFunctions.py
participant = 'S02'
folderpath = 'W:\\CSG\\data\\OPM\\raw'

frequency = 1000

total_path = folderpath + participant 

save_path = total_path + f'/Resampled{frequency}Hz'

isExist = os.path.exists(save_path)


if not isExist:
    # os.makedirs(save_path)
    print('Resampling and saving raw files...')
    ResampleSavefif(folderpath, participant,frequency)
else:
    print(f'The raw files of this participant have already been resampled to {frequency} Hz')
    
    def TC_fixEvents(raw,stimChan):
        """function that adjust analog triggers so they have rounded values"""
     
        data,times = raw.get_data(return_times=True)
        
        idx = raw.info.get('ch_names').index(stimChan)
        
        raw_data = data[idx]
        
        list(raw_data.round(0).astype(int))
        

        
        print(idx)
        
        raw_data=raw_data.round(0).astype(int)
        
        # raw_data[raw_data ==1] = 2
        
        raw._data[idx] = raw_data
     
        return(raw)
