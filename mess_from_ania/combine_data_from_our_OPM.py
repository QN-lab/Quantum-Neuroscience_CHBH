# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:35:22 2023

@author: kowalcau
"""

import pandas as pd
import os.path
import glob
import numpy as np
import functools


# import logging
#Z:\Data\2022_10_21_noise_for_MSL\20221021\1\20221021\1
#base_directory='Z:\\Data\\Processing_directory\\Attenuation_gradiometer\\fake_brain_dBz_only'
base_directory='Z:\\Data\\2023_08_14_bench\\long_runs\\mag_long_run_drift_000\\'
os.chdir(base_directory)

dir_name = os.path.basename(base_directory)
print(dir_name)





os.chdir(base_directory) ##Change to Zurich dir


files = os.listdir(base_directory)

for f in files:
	print(f)
file_name=''#paste here name of the file
#file_name='dev3994_demods_0_sample_'#paste here name of the file

sep=';'
#simplest would be the line below but map by itself does not let you supply additional arguments
#data0=pd.concat(map(pd.read_csv,glob.glob('_stream_shift_avg_0*.csv')), ignore_index= True)
#avg_0* is needed in order to avoid reading in the header files

data0=pd.concat(map(functools.partial(pd.read_csv, sep=sep),glob.glob('dev3994_pids_0_stream_shift_avg_0*.csv')), ignore_index= True)
data1=pd.concat(map(functools.partial(pd.read_csv, sep=sep),glob.glob('dev3994_pids_0_stream_error_avg_0*.csv')), ignore_index= True)
data2=pd.concat(map(functools.partial(pd.read_csv, sep=sep),glob.glob('dev3994_demods_0_sample_auxin0_avg_0*.csv')), ignore_index= True)
data3=pd.concat(map(functools.partial(pd.read_csv, sep=sep),glob.glob('dev3994_demods_0_sample_auxin1_avg_0*.csv')), ignore_index= True)
#data4=pd.concat(map(functools.partial(pd.read_csv, sep=sep),glob.glob('dev3994_demods_0_sample_trigin1_avg_0*.csv')), ignore_index= True)
data5=pd.concat(map(functools.partial(pd.read_csv, sep=sep),glob.glob('dev3994_demods_0_sample_trigin2_avg_0*.csv')), ignore_index= True)
data6=pd.concat(map(functools.partial(pd.read_csv, sep=sep),glob.glob('dev3994_demods_0_sample_x_avg_0*.csv')), ignore_index= True)
data7=pd.concat(map(functools.partial(pd.read_csv, sep=sep),glob.glob('dev3994_demods_0_sample_y_avg_0*.csv')), ignore_index= True)

data0.head(3)

data0['time']=round((data0['timestamp']-data0.iloc[0][1])/60000000,7)

data0['B_T (pT)']=data0['value']*71.488
data0['error_deg']=(data1['value'])
data0['Aux1_v']=(data2['value'])#aux1
data0['Aux1_v']=data0['Aux1_v'].round(0).astype(int)

data0['Aux2_v']=(data3['value'])#aux1
data0['Aux2_v']=data0['Aux2_v'].round(0).astype(int)
#data0['Trig_in1']=(data4['value'])
data0['Trig_in2']=(data5['value']).round(0).astype(int)
data0['Demod_X']=(data6['value'])
data0['Demod_Y']=(data7['value'])
#data0['new_chunk']=()
#data0['Trig_2']=data0['Aux2_v'].round(0).astype(int)


#data0.drop(data0.loc[data0['chunk']==46].index, inplace=True)
# data0['chunk']
nan_values = data0[data0['time'].isna()]
# print(nan_values)
nan_values.shape
data2=data0[data0.isna()]
print(data2)
#data0['Trig_1']=data0['Trig_in1'].round(0).astype(int)


# data2=data0.dropna(axis=0,how='any')


print(data0.head(2))
print(data0.tail(3))
#data0.duplicated(['time'])
data1=data0.drop_duplicates(['timestamp'], keep='last')
#data0.shape
data1.shape
print(data1.head(2))
print(data1.tail(3))
data1= data1.drop('timestamp', axis=1)
#data2.shape
print('################ saving new file ####################')
print('                 please, wait')

#data1.to_csv(file_name+"combined.csv", sep=',', index=False)
data1.to_csv("_f.csv", sep=',', index=False)
