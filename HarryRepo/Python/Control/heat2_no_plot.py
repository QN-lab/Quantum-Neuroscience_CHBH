# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:31:51 2022

@author: H
"""
import numpy as np
import sys
import sounddevice as sd
import serial
from simple_pid import PID
import winsound
import regex as re

try:
#%% Prep I/O
    #Open Adruino serialport
    ard = serial.Serial('COM13',9600)
    ard.close()
    ard.open()
    
    rex1 = '1: (.*\d);;' #Format of arduino output to extract temp values
    rex2 = '2: (.*\d);'
    
    #Determine Audio Output
    sd.default.device = 'Scarlett 2i2 USB MME'
    
    #Prep Sinusoid
    Amp_i = 0.25  #0.5 avoids clipping of max voltage output of the Scarlett
    samplerate = 88200
    start_idx = 0
    freq = 21e3 #Sinusoid Frequency
    
    #Prep Loop
    i = 0 #loop index for time datapoints
    Amp1 = Amp2 = np.array([Amp_i])
    ###########################################################################
#%% Prep PID

    T_targ= [36,40.5]          #Set PID temperature target
    prop =  [0.25, 0.15]     #P 0.4
    integ = [0.01, 0.01]     #I 0.01
    deriv = [0.02, 0.03]     #D # was originally 0.005
    clamp = (0,0.25)         #clamp to 0.5**2
    
    pid1 = PID(prop[0],integ[0],deriv[0],setpoint=T_targ[0])
    pid2 = PID(prop[1],integ[1],deriv[1],setpoint=T_targ[1])
    
    #Clamp, sample rate
    pid1.output_limits = clamp
    pid2.output_limits = clamp
    
    #Prep High Temp Warning
    alarm_dur = 3000
    alarm_freq = 600  #Hz
    alarm_T = 50
    
    ###########################################################################
#%% Functions
    #Perform PID and generate new amplitudes
    def Piding2():
        
        #Temp code to test audio before we do the PID
        data = ard.readline()
        extr1= re.findall(rex1, data.decode())
        extr2 = re.findall(rex2, data.decode())
        try:
            T = np.array([float(extr1[0]),float(extr2[0])])
            
            pid_out1 = pid1(T[0])
            pid_out2 = pid2(T[1])
            
            amp_out = [pid_out1**0.5, pid_out2**0.5] #Power->Current conversion
            
            return amp_out, T
        except:
            T = [0,0]
            amp_out = [0,0]
            print('FAIL')
            winsound.Beep(alarm_freq, alarm_dur)
            return amp_out, T

    #Callback that generates sound sine function
    def callback(outdata, frames: int, time, status):
        global start_idx
        
        #Sine Wave output
        t = (start_idx + np.arange(frames)) / samplerate
        outdata[:,0] = Amp1[-1] * np.sin(2 * np.pi * freq * t)
        outdata[:,1] = Amp2[-1] * np.sin(2 * np.pi * freq * t)
        start_idx += frames
        
    ###########################################################################  
#%% Stream the callback to the Audio Amp
    with sd.OutputStream(channels=2, callback=callback,
                             samplerate=samplerate):
        while True:
            
            amp_out, T = Piding2()
            
            print('T1: {}, T2: {}'.format(T[0],T[1]))
            print('Amp1: {}, Amp2: {}'.format(amp_out[0],amp_out[1]))
            print()
            
            Amp1 = np.append(arr=Amp1[-1], values=amp_out[1])
            Amp2 = np.append(arr=Amp2[-1], values=amp_out[0])
            
            if T[0] >= alarm_T or T[1] >= alarm_T: #High Temperature Warning and exit
                winsound.Beep(alarm_freq, alarm_dur)
                print('################## HIGH TEMP WARNING ##################')
                Amp1=Amp2=0
                ard.close()
                sys.exit('CODE KILLED DUE TO HIGH TEMP')
                
            i += 1
            
except KeyboardInterrupt:
    ard.close() #Close serialport upon keyboard exit
