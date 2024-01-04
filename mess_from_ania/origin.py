# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:18:42 2023

@author: kowalcau
"""

origin_directory='Z:\\Publications_submission\\Gradiometer Paper Prep\\Origin files' # MAKE SURE THERE IS NO \\ AT THE END
os.chdir(origin_directory)
print(data_g.head(1))  

data_gc=data_g.drop(data_g.columns[[1,2,3,4]], axis=1, inplace=False)
print(data_gc.head(1)) 
data_gc.to_csv("g_raw.csv", sep=',', index=False)
data_g_crop=data_g.copy()

data_g_crop.drop(data_g_crop.loc[data_g_crop['Aux2_v']==5].index, inplace=True)
data_g_crop=data_g_crop.drop(data_g_crop.columns[[1,2,3,4]], axis=1, inplace=False)

print(data_g.shape,data_gc.shape,data_g_crop.shape)
data_g_crop.to_csv("gcrop_raw.csv", sep=',', index=False)


data_mc=data_m.drop(data_m.columns[[1,2,3,4]], axis=1, inplace=False)
print(data_mc.head(1)) 
data_mc.to_csv("m_raw.csv", sep=',', index=False)
data_m_crop=data_m.copy()

data_m_crop.drop(data_m_crop.loc[data_m_crop['Aux2_v']==5].index, inplace=True)
data_m_crop=data_m_crop.drop(data_m_crop.columns[[1,2,3,4]], axis=1, inplace=False)

print(data_m.shape,data_mc.shape,data_m_crop.shape)
data_m_crop.to_csv("mcrop_raw.csv", sep=',', index=False)


data_g.iloc[0:10000].plot(x=ch_names[0],y=[ch_names[i] for i in [2,5]])
plt.show()
