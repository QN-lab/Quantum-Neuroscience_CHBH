import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import statistics as stats
import pandas as pd
import Harry_analysis as HCA
import regex as re
import math


loc = 'C:/Users/vpixx/Documents/Zurich Instruments/LabOne/WebServer/session_20230131_121127_08/mag_0.35nT_noise_000/'

spec_m, spec_legends_m = HCA.Spect_read(loc, ';')
        
yes = HCA.Spect_analyser(spec_m, spec_legends_m)

run_names = yes.run_names
frq_domain = yes.frq_domain
# def extract_peaks(run_names,chunked_data):
freq_list_temp = []
freq_list = np.zeros(len(run_names))

for i in range(len(run_names)):
    freq_list_temp = re.findall(r'\d+', run_names[i])
    if not freq_list_temp:
        freq_list_temp = [np.nan]
    freq_list[i] = freq_list_temp[0]


nidx = np.zeros(len(freq_list))
maxval = np.zeros(len(freq_list))


for j in range(len(freq_list)):
    if math.isnan(freq_list[j]):
        pass
    else:
        nidx[j] = (np.abs(frq_domain-freq_list[j])).argmin()
        
        roi = yes.chunked_data[j,int(nidx[j]-5):int(nidx[j]+5)]
        
        maxval[j] = np.max(roi)
        
        

                
    # return freq_list, freq_list_temp, nidx

# q1,q2,nidx = extract_peaks(yes.run_names,yes.chunked_data)
