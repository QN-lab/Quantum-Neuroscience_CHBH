# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:11:38 2023

@author: kowalcau
"""
import numpy as np

hbar=1.05457172533628E-34
muB=9.27400968E-24
Lande_g=0.5

def sensitivity (width, SNR):
    sensitivity=(hbar*width*2*np.pi)/(Lande_g*muB*SNR)*1e15
    return sensitivity

def Hz_to_fT (f_m):
    unit=1e-15
    B_T=round(2*np.pi*f_m*hbar/(2*muB*Lande_g)/unit)
    return B_T

def Hz_to_pT (f_m):
    unit=1e-12
    B_T=round(2*np.pi*f_m*hbar/(2*muB*Lande_g)/unit,3)
    return B_T

def Hz_to_nT (f_m):
    unit=1e-9
    B_T=round(2*np.pi*f_m*hbar/(2*muB*Lande_g)/unit,3)
    return B_T

def mHz_to_fT (f_m):
    unit=1e-15
    B_T=round(2*np.pi*f_m*hbar*1e-3/(2*muB*Lande_g)/unit,3)
    return B_T