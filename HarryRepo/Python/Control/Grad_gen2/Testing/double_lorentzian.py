# # Log sequence recorded on 2024/04/24 13:15:56

###############################################################################
###############################################################################

from zhinst.toolkit import Session
import logging
import sys
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.optimize import differential_evolution
plt.close('all')

session = Session("localhost")
device = session.connect_device("DEV3994")

OUT_CHANNEL = 0         #Output channel: Sig out 1
OUT_MIXER_CHANNEL = 1   # UHFLI: 3, HF2LI: 6, MFLI: 1
IN_CHANNEL = 1          #0:
CURR_INDEX = 0
DEMOD_INDEX = 0         #Demodulator Index
OSC_INDEX = 0           #Oscillator index (I think we only have 1 oscillator)

BIAS_OFFSET_0 = 0.320
BIAS_OFFSET_1 = 0.260

#SET NODES
with device.set_transaction():
  
    device.currins[CURR_INDEX].on(True)
    device.currins[CURR_INDEX].autorange()
    device.currins[CURR_INDEX].float(0)                           #Floating ground    OFF
    device.currins[CURR_INDEX].scaling(1)                         #Input scaling      1V/1V
    device.currins[CURR_INDEX].range(0.000010)   
    # device.currins[IN_CHANNEL].range(0.010)                     #Amplifier range    

    # assert device.sigins[IN_CHANNEL].range() <= 0.30000001192092896, (
    #     'Autorange has not settled to the minimum value: check sensor and DC offsets'
    #     )
        
    device.demods[DEMOD_INDEX].adcselect(IN_CHANNEL)            #Select ADC          
    device.demods[DEMOD_INDEX].order(4)                         #LPF order
    device.demods[DEMOD_INDEX].rate(13.39e3)                    #PC sampling rate for data collection
    device.demods[DEMOD_INDEX].oscselect(OSC_INDEX)             #select internal oscillator (only one for us?)
    device.demods[DEMOD_INDEX].harmonic(1)                      #Multiplies ref osc freq by this integer factor (INVESTIGATE NOTE IN THIS ABOUT PLL LOCKING)
    device.demods[DEMOD_INDEX].phaseshift(-108)                 #Applied phase shift (deg) to internal ref
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
    
    
    #Set Aux channels
    device.auxouts[0].outputselect(4)
    device.auxouts[0].offset(BIAS_OFFSET_0)
    
    device.auxouts[1].outputselect(4)
    device.auxouts[1].offset(BIAS_OFFSET_1)

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
    sweeper.bandwidthcontrol(1.5) #Changes bandwidth
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
def Lorentzian_double(x,amp0,amp1,cen0,cen1,wid0,wid1,slope,offset):
    return ((amp0*(wid0)**2/((x-cen0)**2+(wid0)**2)) + (amp1*(wid1)**2/((x-cen1)**2+(wid1)**2))) +slope*x + offset

def Lorentzian_single(x,amp0,amp1,cen0,cen1,wid0,wid1,slope,offset):
    return ((amp0*(wid0)**2/((x-cen0)**2+(wid0)**2)) + (amp1*(wid1)**2/((x-cen1)**2+(wid1)**2))) +slope*x + offset
###############################################################################
#Fitting
def L_fit_double(func,freq,x):
    popt_lor, pcov_lor = curve_fit(func,freq,x,p0=[0.1,0.1,2.5e3,3e3,50,50,0,0]) 
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
    axs[0].set_ylabel("Quadrature signal (Amps)")
    # axs[0].set_xscale("log")
    axs[0].autoscale()
    
    axs[1].grid()
    axs[1].set_xlabel("Frequency ($Hz$)")
    axs[1].set_ylabel("In-phase Signal (Amps)")
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
# function for genetic algorithm to minimize (sum of squared error) CITATION
# bounds on parameters are set in generate_Initial_Parameters() below
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    return np.sum((yData - Lorentzian_double(xData, *parameterTuple)) ** 2)

def generate_Initial_Parameters():
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)
    
    parameterBounds = []
    
    parameterBounds.append([0.0, maxY*2]) # parameter bounds for A
    parameterBounds.append([0.0, maxY*2]) # parameter bounds for A1
    parameterBounds.append([minX, maxX]) # parameter bounds for x_0
    parameterBounds.append([minX, maxX]) # parameter bounds for x_01
    parameterBounds.append([0.0, 150]) # parameter bounds for w
    parameterBounds.append([0.0, 150]) # parameter bounds for w1
    parameterBounds.append([-0.005, 0.005]) # parameter bounds for slope
    parameterBounds.append([maxY/-0.005, maxY/0.005]) # parameter bounds for offset

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

###############################################################################

data, sample_node = sweep_now(device,1.9e3,3.5e3,300,OSC_INDEX,DEMOD_INDEX,0)

node_samples = data[sample_node] #extract data from dict

xData = node_samples[0][0]["frequency"] #Frequency values from sweeper
yData = node_samples[0][0]["x"]

initialParameters = generate_Initial_Parameters()

# curve fit the test data
fittedParameters, pcov = curve_fit(Lorentzian_double, xData, yData, initialParameters)

# #Plot swept data
fig, axs = plot_sweep(node_samples,sample_node)

A0,A1,c0,c1,w0,w1,s,o = fittedParameters
fit = Lorentzian_double(xData,A0,A1,c0,c1,w0,w1,s,o)

axs[0].plot(xData, fit, ls ='--',c='red')

axs[0].plot(xData,(s*xData+o),ls ='--',c='green')
###############################################################################
#Value Report
print('############################################################')
print('Fit Parameters')
print('Amplitudes(nA): ' + str(round(A0*1e9,5))+'; '+str(round(A1*1e9,5)))
print('Central freq(Hz): ' + str(round(c0,0))+'; '+str(round(c1,0)))
print('FWHM(Hz): ' + str(round(w0*2,0))+'; '+str(round(w1*2,0)))
print('Slope(nA/Hz): ' + str(round(s*1e9,10)))
print('0Hz intercept(nA): ' + str(round(o*1e9,5)))
print('############################################################')


#If one fitted lorentzian is far larger than the other then we have a signle lorentzian. 
if A0 > A1*4 or A0*4 < A1 or w0>w1*1.5 or w0*1.5 < w1:
    print('Single Lorentzian detected')

###############################################################################
#Fit linear range

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

inpData = node_samples[0][0]["y"]

SearchRangeAll = [c0-w0/4, c0+w0/4, c1-w1/4, c1+w1/4] # Goes from left side of one resonance to right side of the other. 

SearchRange = [min(SearchRangeAll),max(SearchRangeAll)]

RangeWidth = SearchRange[1]-SearchRange[0]

print('Search range size: '+ str(round(RangeWidth))+' Hz or ' 
      + str(round(RangeWidth/14.2,1))+'nT')

IndexRange = [find_nearest(xData,SearchRange[0]),find_nearest(xData,SearchRange[1])]

fLin = xData[IndexRange[0]:IndexRange[1]]
inpLin = inpData[IndexRange[0]:IndexRange[1]]

axs[1].axvline(x=SearchRange[0],ls='-', c='cyan')
axs[1].axvline(x=SearchRange[1],ls='-', c='cyan')

#Fit a poly and a line to compare linearity of response
fig2, ax2 = plt.subplots()
ax2.plot(fLin,inpLin,'o')

pPoly= np.polyfit(fLin, inpLin, 3) #Extract p values from 5-degree poly
fittedpoly = np.poly1d(pPoly)

ax2.plot(fLin,fittedpoly(fLin),linewidth=2.5,label='Polyfit x$^3$ coef = '+ str(round(pPoly[0],20)))

pLin = np.polyfit(fLin, inpLin, 1)
fittedLin = np.poly1d(pLin)

slopeLin = pLin[0]
x3coef = pPoly[0]

ax2.plot(fLin,fittedLin(fLin),linewidth=2.5,label='LinFit Linslope= '+ str(round(slopeLin,14)))

ax2.set_xlabel('Frequency ($Hz$)')
ax2.set_ylabel('In-Phase Signal (Amp)')
ax2.set_title('Linearity report in ROI = ' + str(round(RangeWidth)) + ' Hz')
ax2.legend()

print('x$^3$ coeff = ' + str(x3coef))





