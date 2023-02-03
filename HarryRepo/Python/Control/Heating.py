# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:57:59 2022

@author: kowalcau-admin
"""
import numpy as np
import sounddevice as sd
import serial
from simple_pid import PID
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import winsound


#Need To change x axis on plots to time. Sample rate of the PID doesn't seem constant
#   This can be changed but it may screw with the loop, we shall see.

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
    Amp_i = 0.25  #0.5 avoids clipping of max voltage
    samplerate = 44100
    start_idx = 0
    freq1 = 1e3
    freq2 = 5e3
    
    #Prep PID
    T_targ= 32.0 #Set PID temperature target
    prop = 0.1
    integ = 0.0002
    deriv = 0.0002 # was 0.0001
    pid = PID(prop,integ,deriv,setpoint=T_targ)
    #Clamp and limit
    pid.output_limits = (0, 0.25) #sqrt of 0.25 is 0.5, our max output
    
    #Prep High Temp Warning
    alarm_dur = 3000
    alarm_freq = 600  # Hz
    alarm_T = 45
    
    #Prep Loop
    saveindx = 20 #Avoids filling RAM for long runs
    i = 0 #loop index for time datapoints
    
    #Plotting
    x = y = pid_plot = np.array([]) #plot params
    P = I = D = np.array([]) #track PID values
    Amp1 = Amp2 = np.array([Amp_i])
    
    fig1=plt.figure()
    ax = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)
    
    fig2 = plt.figure()
    Pax = fig2.add_subplot(311)
    Iax = fig2.add_subplot(312)
    Dax = fig2.add_subplot(313)
    
    #Perform PID and generate new amplitudes
    def Piding():
        
        #Temp code to test audio before we do the PID
        data = ard.readline()
        T_live = float(data.decode())
        pid_out = pid(T_live)
        P,I,D = pid.components
        
        map_out = pid_out**0.5 #Power is proportional to current squared
        
        return map_out, T_live, P, I, D #Amp1,Amp2

    #Callback that generates sound sine function
    def callback(outdata, frames: int, time, status):
        
        global start_idx
        
        #Sine Wave output
        t = (start_idx + np.arange(frames)) / samplerate
        outdata[:,0] = Amp1[-1] * np.sin(2 * np.pi * freq1 * t)
        outdata[:,1] = Amp2[-1] * np.sin(2 * np.pi * freq2 * t)
        start_idx += frames
        
    #Stream the callback to the Audio Amp
    with sd.OutputStream(channels=2, callback=callback,
                             samplerate=samplerate):
        while True:
            
            map_out, T, Pi, Ii, Di = Piding()
            
            print('Mapped PID Output:   ' + str(map_out))
            print('Temp:         ' + str(T))
            print()
            
            #Plot Live Temperature
            x = np.append(arr=x,values=i)
            y = np.append(arr=y,values=T)
            pid_plot = np.append(arr=pid_plot, values=map_out)
            
            xPID = x
            #Recalculate Amplitudes
            
            Amp1 = np.append(arr=Amp1[-1], values=map_out)
            Amp2 = np.append(arr=Amp2[-1], values=map_out) 
            
            if len(x) >= saveindx:
                x = x[-saveindx:]
                y = y[-saveindx:]
                pid_plot = pid_plot[-saveindx:]
                
            if len(P) > saveindx: 
                P = P[-saveindx:]
                I = I[-saveindx:]
                D = D[-saveindx:]
                xPID = np.append(arr=x,values=x[-1]+1)
            
            
            if T >= alarm_T: #High Temperature Warning and exit
                winsound.Beep(alarm_freq, alarm_dur)
                print('################## HIGH TEMP WARNING ##################')
                Amp1=Amp2=0
                exit()
                
            line1, = ax.plot(x,y,'b-')
            line2, = ax2.plot(x,pid_plot,'k-')
            
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            fig1.show()
            
            #Plotting PID components
            P = np.append(arr=P,values=Pi)
            I = np.append(arr=I,values=Ii)
            D = np.append(arr=D,values=Di)
            
            Pline, = Pax.plot(xPID,P,'k-')
            Iline, = Iax.plot(xPID,I,'k-')
            Dline, = Dax.plot(xPID,D,'k-')
            
            fig2.canvas.draw()
            fig2.canvas.flush_events()
            fig2.show()
            i += 1
            
except KeyboardInterrupt:
    ard.close()

