# -*- coding: utf-8 -*-
import pandas as pd

#Speed up data read-in
def ReadData(folder_path,filename,headername,csv_sep):
    out_sig = pd.read_csv(folder_path+filename,sep=csv_sep)
    out_headers = pd.read_csv(folder_path+headername,sep=csv_sep)
    return out_sig, out_headers


def Res_read(folder_path,csv_sep):
    filename = 'dev3994_demods_0_sample_00000.csv'
    headername = 'dev3994_demods_0_sample_header_00000.csv'
    
    out_sig, out_headers = ReadData(folder_path, filename, headername, csv_sep)
    
    return out_sig, out_headers

def Spect_read(folder_path,csv_sep):
    filename = 'dev3994_demods_0_sample_xiy_fft_abs_pwr_avg_00000.csv'
    headername = 'dev3994_demods_0_sample_xiy_fft_abs_pwr_avg_header_00000.csv'

    out_sig, out_headers = ReadData(folder_path, filename, headername, csv_sep)

    return out_sig, out_headers

def DAQ_spect_read(folder_path,csv_sep):
    filename = 'dev3994_demods_0_sample_xiy_fft_abs_avg_00000.csv' #Spectrum Data
    headername =  'dev3994_demods_0_sample_xiy_fft_abs_avg_header_00000.csv'
    
    out_sig, out_headers = ReadData(folder_path, filename, headername, csv_sep)
    
    return out_sig, out_headers
    
def DAQ_tracking_read(folder_path,csv_sep):
    filename = 'dev3994_demods_0_sample_frequency_avg_00000.csv'
    headername = 'dev3994_demods_0_sample_frequency_avg_header_00000.csv' 
    
    out_sig, out_headers = ReadData(folder_path, filename, headername, csv_sep)
    
    return out_sig, out_headers

def DAQ_trigger_read(folder_path,csv_sep):
    filename = 'dev3994_demods_0_sample_trigin2_avg_00000.csv'
    headername = 'dev3994_demods_0_sample_trigin2_avg_header_00000.csv'
    
    out_sig, out_headers = ReadData(folder_path, filename, headername, csv_sep)
    
    return out_sig, out_headers

def DAQ_read_auxin0(folder_path,csv_sep):

    filename = 'dev3994_demods_0_sample_auxin0_avg_00000.csv'
    headername = 'dev3994_demods_0_sample_auxin0_avg_header_00000.csv'
    
    out_sig, out_headers = ReadData(folder_path, filename, headername, csv_sep)
    
    return out_sig, out_headers


def DAQ_read_shift(folder_path,csv_sep):
    
    filename = 'dev3994_pids_0_stream_shift_avg_00000.csv'
    headername = 'dev3994_pids_0_stream_shift_avg_header_00000.csv'

    out_sig, out_headers = ReadData(folder_path, filename, headername, csv_sep)

    return out_sig, out_headers

def limitise(data,fraction):
    bottom = min(data)-fraction*abs(min(data))
    top = max(data)+fraction*abs(max(data))
    lims = (bottom,top)
    return lims

def limitise_werr(data,error,fraction):
    bottom = min(data)-abs(max(error))-fraction*abs(min(data))
    top = max(data)+abs(max(error))+fraction*abs(max(data))
    lims = (bottom,top)
    return lims


    