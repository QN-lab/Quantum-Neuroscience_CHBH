# -*- coding: utf-8 -*-
"""
8-12-22

@author: Harry
"""
import numpy as np
import sounddevice as sd
import serial
from simple_pid import PID
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import winsound
import regex as re

#Need to add axes labels etc to plots. 

try:
    #Open Adruino serialport
    ard = serial.Serial('COM8',9600)
    ard.close()
    ard.open()
    
    #Determine Audio Output
    sd.default.device = 'Scarlett 2i2 USB MME'
    # sd.default.device = 'Headphones MME'
    
    #Prep Sinusoid
    Amp_i = 0.25 # start off the PID
    samplerate = 44100
    start_idx = 0
    freq = [50e3,50e3]
    
    #Prep PIDs
    
    T_targ = [35, 39] # Cell1 (brain), Cell2
    prop = [0.1, 0.1]
    integ = [0.0002, 0.0002]
    deriv = [0.0002, 0.0002]
    clamp = (0,0.25) #sqrt of 0.25 is 0.5, our max output
    
    #PID1
    pid1 = PID(prop[0],integ[0],deriv[0],setpoint=T_targ[0])
    pid2 = PID(prop[1],integ[1],deriv[1],setpoint=T_targ[1])
    
    pid1.output_limits = clamp    
    pid2.output_limits = clamp

    #Prep High Temp Warning
    alarm_dur = 3000
    alarm_freq = 600  # Hz
    alarm_T = 50
    
    #Prep Loop
    saveindx = 20 #should hopefully reduce RAM use for longer runs 
    i = 0 #loop index for time datapoints
    
    #Plotting
    x = y1 = y2 = pid_plot1 = pid_plot2 = np.array([]) #plot params
    PID_vals = np.zeros((6, 1))
    Amp1 = Amp2 = np.array([Amp_i])
    
    #Plot temp and Driving Amplitude from PIDs
    fig1=plt.figure()
    ax11 = fig1.add_subplot(221) # PID 1
    ax12 = fig1.add_subplot(222)
    
    ax21 = fig1.add_subplot(223) # PID 2
    ax22 = fig1.add_subplot(224)
    
    #Plot PID values for both PIDs
    fig2 = plt.figure()
    Pax1 = fig2.add_subplot(611) # PID 1
    Iax1 = fig2.add_subplot(612)
    Dax1 = fig2.add_subplot(613)
    
    Pax2 = fig2.add_subplot(614) # PID 2
    Iax2 = fig2.add_subplot(615)
    Dax2 = fig2.add_subplot(616)
    
    ###############################################################################################
    #%% Functions for streaming
        #Perform PID and generate new amplitudes
    reg1 = 'T1: (.*\d);;'
    reg2 = 'T2: (.*\d);'
    def Piding2():
        #Out:   Amp - 1x2
        #       T - 1x2
        #       PID_vals - 1x6
        T = np.array([])
        #Temp code to test audio before we do the PID
        data = ard.readline()
        
        readout = data.decode()
        T1_str = re.findall(reg1, readout)
        T2_str = re.findall(reg2, readout)
        T = np.array([float(T1_str[0]),float(T2_str[0])])
        
        pid_out1 = pid1(T[0])
        pid_out2 = pid2(T[1])
        
        amp_out = [pid_out1**0.5, pid_out2**0.5] #Power is proportional to current squared
        
        P1,I1,D1 = pid1.components
        P2,I2,D2 = pid2.components
        PID_valsi = np.array([[P1],[I1],[D1],[P2],[I2],[D2]])

        return amp_out, T, PID_valsi
    ###############################################################################################
    #Callback that generates sound sine function
    def callback(outdata, frames: int, time, status):
        
        global start_idx
        
        #Sine Wave output
        t = (start_idx + np.arange(frames)) / samplerate
        outdata[:,0] = Amp1[-1] * np.sin(2 * np.pi * freq[0] * t) #[Change to Amp[0,0]??
        outdata[:,1] = Amp2[-1] * np.sin(2 * np.pi * freq[1] * t)
        start_idx += frames
        
    ###############################################################################################
    #%% Stream    
    #Stream the callback to the Audio Amp, whilst calling the PID
    with sd.OutputStream(channels=2, callback=callback,
                             samplerate=samplerate):
        while True:
            
            amp_out,T, PID_valsi = Piding2()
            
            print()
            print('Temp:         C1:  {}C, C2:  {}C'.format(T[0],T[1]))
            print()
            
            #Plot Live Temperature
            x = np.append(arr=x,values=i)
            y1 = np.append(arr=y1,values=T[0])
            y2 = np.append(arr=y2,values=T[1])
            
            pid_plot1 = np.append(arr=pid_plot1, values=amp_out[0])
            pid_plot2 = np.append(arr=pid_plot2, values=amp_out[1])
            
            PID_vals = np.hstack((PID_vals,PID_valsi))

            if PID_vals[0,0] == 0:
                PID_vals = PID_vals[:,1:]
            
            #Adjusting Amplitude
            Amp1 = np.append(arr=Amp1[-1], values=amp_out[0])
            Amp2 = np.append(arr=Amp2[-1], values=amp_out[1])
            
            if len(x) >= saveindx:
                x = x[-saveindx:]
                y1 = y1[-saveindx:]
                y2 = y2[-saveindx:]
                pid_plot1 = pid_plot1[-saveindx:]
                pid_plot2 = pid_plot2[-saveindx:]
                
            if PID_vals.shape[1] > saveindx: #CHANGE FOR 6
                PID_vals = PID_vals[:,-saveindx:]
            
            if T[0] >= alarm_T or T[1] >= alarm_T: #High Temperature Warning and exit
                winsound.Beep(alarm_freq, alarm_dur)
                print('################## HIGH TEMP WARNING ##################')
                Amp1=Amp2=0
                exit()
            
            #######################################################################################
            #Plotting
            
            line11, = ax11.plot(x,y1,'b-')
            line12, = ax12.plot(x,pid_plot1,'k-')
            
            line21, = ax21.plot(x,y2,'b-')
            line22, = ax22.plot(x,pid_plot2,'k-')
            
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            fig1.show()
            
            Pline1, = Pax1.plot(x,PID_vals[0,:],'k-')
            Iline1, = Iax1.plot(x,PID_vals[1,:],'k-')
            Dline1, = Dax1.plot(x,PID_vals[2,:],'k-')
            
            Pline2, = Pax2.plot(x,PID_vals[3,:],'k-')
            Iline2, = Iax2.plot(x,PID_vals[4,:],'k-')
            Dline2, = Dax2.plot(x,PID_vals[5,:],'k-')
            
            fig2.canvas.draw()
            fig2.canvas.flush_events()
            fig2.show()
            
            i += 1
            
except KeyboardInterrupt:
    ard.close()

