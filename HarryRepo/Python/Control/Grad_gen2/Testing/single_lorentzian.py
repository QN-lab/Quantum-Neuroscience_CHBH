# # Log sequence recorded on 2024/04/24 13:15:56
# ###############################################################################
# ###############################################################################

from toptica.lasersdk.dlcpro.v3_2_0 import DLCpro,NetworkConnection
from zhinst.toolkit import Session
import logging
import sys
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.optimize import differential_evolution
import serial
import time

session = Session("localhost")
device = session.connect_device("DEV3994")

OUT_CHANNEL = 0         #Output channel: Sig out 1
OUT_MIXER_CHANNEL = 1   # UHFLI: 3, HF2LI: 6, MFLI: 1
IN_CHANNEL = 1          #0:
CURR_INDEX = 0
DEMOD_INDEX = 0         #Demodulator Index
OSC_INDEX = 0           #Oscillator index (I think we only have 1 oscillator)

#SET NODES
with device.set_transaction():
  
    device.currins[CURR_INDEX].on(True)
    device.currins[CURR_INDEX].autorange()
    device.currins[CURR_INDEX].float(0)                           #Floating ground    OFF
    device.currins[CURR_INDEX].scaling(1)                         #Input scaling      1V/1V
    device.currins[CURR_INDEX].range(0.000100)   
    device.currins[CURR_INDEX].autorange(1)                       #Amplifier range    

    # assert device.sigins[CURR_INDEX].range() < 10e-06, (
    #     'Autorange has not settled to the minimum value: check sensor and DC offsets'
    #     )
        
    device.demods[DEMOD_INDEX].adcselect(IN_CHANNEL)            #Select ADC          
    device.demods[DEMOD_INDEX].order(4)                         #LPF order
    device.demods[DEMOD_INDEX].rate(13.39e3)                    #PC sampling rate for data collection
    device.demods[DEMOD_INDEX].oscselect(OSC_INDEX)             #select internal oscillator (only one for us?)
    device.demods[DEMOD_INDEX].harmonic(1)                      #Multiplies ref osc freq by this integer factor (INVESTIGATE NOTE IN THIS ABOUT PLL LOCKING)
    device.demods[DEMOD_INDEX].phaseshift(72)                   #Applied phase shift (deg) to internal ref
    device.demods[DEMOD_INDEX].sinc(0)                          #Sinc filter OFF
    device.demods[DEMOD_INDEX].timeconstant(0.0007)             #Filter time constant
    device.demods[DEMOD_INDEX].enable(True)                     #Enables Data collection for this demodulator, increases load on PC port
    
    device.sigouts[OUT_CHANNEL].on(True)                        #Output                     ON
    device.sigouts[OUT_CHANNEL].imp50(1)                        #50Ohm Imp                  ON 
    device.sigouts[OUT_CHANNEL].diff(0)                         #Single output; diff output OFF
    device.sigouts[OUT_CHANNEL].range(1)                        #Range 5V
    device.sigouts[OUT_CHANNEL].offset(0)                       #DC offset 0V
    device.sigouts[OUT_CHANNEL].enables[OUT_MIXER_CHANNEL](1)   #Enable output amplitudes to drive to sig out 1 (switch 'En' on Block Diagram)
    device.sigouts[OUT_CHANNEL].amplitudes(3)                   #Output amplitude max Vpk
    
###############################################################################

#Sweeper function
def sweep_now(device,start,stop,samples,OSC_INDEX,DEMOD_INDEX,save):
    # Specify the number of sweeps to perform back-to-back.
    LOOPCOUNT = 1
    
    sweeper = session.modules.sweeper
    sweeper.device(device)
    
    sweeper.gridnode(device.oscs[OSC_INDEX].freq)
    sweeper.start(start)
    sweeper.stop(stop)
    sweeper.samplecount(samples)
    sweeper.xmapping(1)
    sweeper.bandwidthcontrol(1.5)
    sweeper.bandwidthoverlap(0)
    sweeper.scan(0)
    sweeper.loopcount(LOOPCOUNT)
    sweeper.settling.time(0)
    sweeper.settling.inaccuracy(0.001)
    sweeper.averaging.tc(0)
    sweeper.averaging.sample(1)
    
    sample_node = device.demods[DEMOD_INDEX].sample
    sweeper.subscribe(sample_node)
    
    sweeper.save.filename('sweep_with_save')
    sweeper.save.fileformat('csv')
    
    handler = logging.StreamHandler(sys.stdout)
    logging.getLogger("zhinst.toolkit").setLevel(logging.INFO)
    logging.getLogger("zhinst.toolkit").addHandler(handler)
    
    sweeper.execute()
    print(f"Perform {LOOPCOUNT} sweeps")
    sweeper.wait_done(timeout=300)
    
    data = sweeper.read()
    sweeper.unsubscribe(sample_node)
    num_sweeps = len(data[sample_node])
    assert num_sweeps == LOOPCOUNT, (
        f"The sweeper returned an unexpected number of sweeps: "
        f"{num_sweeps}. Expected: {LOOPCOUNT}."
    )
    
    if save==1:
        sweeper.save.save(True)
        # Wait until the save is complete. The saving is done asynchronously in the background
        # so we need to wait until it is complete. In the case of the sweeper it is important
        # to wait for completion before before performing the module read command. The sweeper has
        # a special fast read command which could otherwise be executed before the saving has
        # started.
        sweeper.save.save.wait_for_state_change(True, invert=True, timeout=5)
        
        print("SAVED DATA")
        
        return data, sample_node
    else:
        
        return data, sample_node

###############################################################################
#Functions to fit
def Lorentzian(x, amp, cen, wid, slope, offset):
    return ((amp*(wid)**2/((x-cen)**2+(wid)**2)) + slope*x + offset)

###############################################################################
#Fitting
def L_fit(func,freq,x):
    popt_lor, pcov_lor = curve_fit(func,freq,x,p0=[0.2e-9,3e3,150,0,0])

    return popt_lor, pcov_lor


###############################################################################
#Plot swept data
def plot_sweep(node_samples,sample_node):
    
    fig, axs = plt.subplots(3, 1)
    for sample in node_samples:
        frequency = sample[0]["frequency"]
        x = sample[0]["x"]
        y = sample[0]["y"]
        phi = np.angle(sample[0]["x"] + 1j * sample[0]["y"])
        
        axs[0].plot(frequency, x)
        axs[1].plot(frequency, y)
        axs[2].plot(frequency, phi)
        
    axs[0].set_title(f"Results of {len(node_samples)} sweeps.")
    axs[0].grid()
    axs[0].set_ylabel("Quadrature signal (nA)")
    # axs[0].set_xscale("log")
    axs[0].autoscale()
    
    axs[1].grid()
    axs[1].set_xlabel("Frequency ($Hz$)")
    axs[1].set_ylabel("In-phase Signal (mV)")
    # axs[1].set_xscale("log")
    axs[1].autoscale()
    
    axs[2].grid()
    axs[2].set_xlabel("Frequency ($Hz$)")
    axs[2].set_ylabel(r"Phi (radians)")
    # axs[2].set_xscale("log")
    axs[2].autoscale()
    
    plt.draw()
    plt.show()
    
    return fig, axs

###############################################################################

data, sample_node = sweep_now(device,1.9e3,3.5e3,150,OSC_INDEX,DEMOD_INDEX,0)

node_samples = data[sample_node] #extract data from dict

f = node_samples[0][0]["frequency"] #Frequency values from sweeper
x = node_samples[0][0]["x"]


fit_params ,fit_cov= L_fit(Lorentzian,f,x) #Fit to one lorentzian

amp =np.array([fit_params[0]])      #Millivolt
centr = np.array([fit_params[1]])   #Central Frequency
wid = np.array([fit_params[2]])     #HWHM (HALF-WIDTH)
slope = np.array([fit_params[3]])   #slope
offset = np.array([fit_params[4]])  #offset

# # #Plot swept data
fig, axs = plot_sweep(node_samples,sample_node)

fit = Lorentzian(f,amp[0],centr[0],wid[0],slope[0],offset[0])

axs[0].plot(f, fit, ls ='--',c='red')

axs[0].plot(f,(slope[0]*f+offset[0]),ls ='--',c='green')


