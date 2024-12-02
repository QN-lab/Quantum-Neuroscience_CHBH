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
OSC_INDEX = 0           #Oscillator isndex (I think we only have 1 oscillator)
  
BIAS_OFFSET_0 = 0.310
BIAS_OFFSET_1 = 0.250

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
    device.demods[DEMOD_INDEX].phaseshift(72)                 #Applied phase shift (deg) to internal ref
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
###############################################################################
#Sweeper function
def sweep_now(device,start,stop,samples,OSC_INDEX,DEMOD_INDEX,save,iteration):
    # Specify the number of sweeps to perform back-to-back.
    LOOPCOUNT = 1
    
    sweeper = session.modules.sweeper
    sweeper.device(device)
    
    sweeper.gridnode(device.oscs[OSC_INDEX].freq)
    sweeper.start(start)
    sweeper.stop(stop)
    sweeper.samplecount(samples)
    sweeper.xmapping(1)
    sweeper.bandwidthcontrol(1) #Changes bandwidth
    sweeper.bandwidthoverlap(0)
    sweeper.scan(0)
    sweeper.loopcount(LOOPCOUNT)
    sweeper.settling.time(0)
    sweeper.settling.inaccuracy(0.001)
    sweeper.averaging.tc(0)
    sweeper.averaging.sample(1)
    
    sample_node = device.demods[DEMOD_INDEX].sample
    sweeper.subscribe(sample_node)
    
    # sweeper.save.filename('sweep_with_save')
    # sweeper.save.fileformat('csv')
    
    handler = logging.StreamHandler(sys.stdout)
    logging.getLogger("zhinst.toolkit").setLevel(logging.WARNING)
    logging.getLogger("zhinst.toolkit").addHandler(handler)
    
    sweeper.execute()
    print(f"Sweeping no. {iteration}")
    sweeper.wait_done(timeout=300)
    
    data = sweeper.read()
    sweeper.unsubscribe(sample_node)
    num_sweeps = len(data[sample_node])
    assert num_sweeps == LOOPCOUNT, (
        f"The sweeper returned an unexpected number of sweeps: "
        f"{num_sweeps}. Expected: {LOOPCOUNT}."
    )
    return data, sample_node

###############################################################################
###############################################################################
#Functions to fit
def Lorentzian_double(x,amp0,amp1,cen0,cen1,wid0,wid1,slope,offset):
    return ((amp0*(wid0)**2/((x-cen0)**2+(wid0)**2)) + (amp1*(wid1)**2/((x-cen1)**2+(wid1)**2))) +slope*x + offset

def Lorentzian_single(x,amp0,cen0,wid0,slope,offset):
    return ((amp0*(wid0)**2/((x-cen0)**2+(wid0)**2))) +slope*x + offset
###############################################################################    
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
###############################################################################
# function for genetic algorithm to minimize (sum of squared error) CITATION
# bounds on parameters are set in generate_Initial_Parameters() below

#Double Lorentzian
def sumOfSquaredError_double(parameterTuple, xData, yData):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    return np.sum((yData - Lorentzian_double(xData, *parameterTuple)) ** 2)

def generate_Initial_Parameters_double(xData, yData):
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)
    
    parameterBounds = [
        [maxY/1.5, maxY*2],    # parameter bounds for A
        [maxY/1.5, maxY*2],    # parameter bounds for A1
        [minX, maxX],     # parameter bounds for c0
        [minX, maxX],     # parameter bounds for c1
        [10, 150],       # parameter bounds for w0 (HWHM)
        [10, 150],       # parameter bounds for w1 
        [-0.005, 0.005],  # parameter bounds for slope
        [maxY/-0.005, maxY/0.005]  # parameter bounds for offset
    ]
    
    # if parameterBounds[1][0]>parameterBounds[1][1]:
    #     parameterBounds
        

    def sumOfSquaredError_wrapper_double(parameterTuple):
        return sumOfSquaredError_double(parameterTuple, xData, yData)

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError_wrapper_double, parameterBounds, seed=3)
    return result.x

def sumOfSquaredError_single(parameterTuple, xData, yData):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    return np.sum((yData - Lorentzian_single(xData, *parameterTuple)) ** 2)

def generate_Initial_Parameters_single(xData, yData):
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)
    
    parameterBounds = [
       [maxY/1.5, maxY*2],    # parameter bounds for A
       [minX, maxX],     # parameter bounds for c0
       [10, 150],       # parameter bounds for w0 (HWHM)
       [-0.005, 0.005],  # parameter bounds for slope
       [maxY/-0.005, maxY/0.005]  # parameter bounds for offset
       ]

    def sumOfSquaredError_wrapper_single(parameterTuple):
        return sumOfSquaredError_single(parameterTuple, xData, yData)

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError_wrapper_single, parameterBounds, seed=3)
    return result.x

###############################################################################
###############################################################################
def read_and_fit(data,sample_node,num):
    
    node_samples = data[sample_node] #extract data from dict

    xData = node_samples[0][0]["frequency"] #Frequency values from sweeper
    yData = node_samples[0][0]["x"]
    
    if num == 1:
        initialParameters = generate_Initial_Parameters_single(xData,yData)
        
        # curve fit the test data
        fittedParameters, _ = curve_fit(Lorentzian_single, xData, yData, initialParameters)
        fit_eval = Lorentzian_single(xData,*fittedParameters)
    elif num == 2:
        initialParameters = generate_Initial_Parameters_double(xData,yData)
    
        # curve fit the test data
        fittedParameters, _ = curve_fit(Lorentzian_double, xData, yData, initialParameters)
        fit_eval = Lorentzian_double(xData,*fittedParameters)
    
    return xData, yData, fittedParameters, fit_eval
###############################################################################
###############################################################################
#Coarse sweep

start_b = 0.220
stop_b = 0.340
num = 5

print(f'Performing Coarse Bias search. Number of sweeps: {num}')

biases = np.linspace(start_b,stop_b,num=num)

perr_list = np.zeros((8,len(biases)))
param_list = np.zeros((8,len(biases)))

r_squared_list = list()

fig, axs = plt.subplots()

for i, bias in enumerate(biases): 
    
    device.auxouts[1].outputselect(4)
    device.auxouts[1].offset(bias)
    
    data, sample_node = sweep_now(device,1.4e3,3.5e3,100,OSC_INDEX,DEMOD_INDEX,0,i+1)

    xData, yData, fittedParameters, fit_eval = read_and_fit(data,sample_node,2)

    param_list[:,i] = fittedParameters

    residuals = yData - Lorentzian_double(xData,*fittedParameters)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yData-np.mean(yData))**2)
    r_squared = 1-(ss_res/ss_tot)
    r_squared_list.append(r_squared)
    
    axs.plot(xData,yData,'o',markersize = 0.5,label='run' + str(i))
    axs.plot(xData,fit_eval,'--',c='blue')
    plt.pause(0.005)
    

r_squared_list = np.array(r_squared_list)
#R squared is largest where overlap is best!

###############################################################################
###############################################################################
#%% 
#Swap to single-Lorentzian and optimise amplitude/width around this region.

print('Fitting Single Lorentzian')

opt_inx = np.argmax(r_squared_list) #Find best run
bias = biases[opt_inx]              #Re-adjust bias to this bias value

device.auxouts[1].outputselect(4)
device.auxouts[1].offset(bias)
data, sample_node = sweep_now(device,1.6e3,3.2e3,150,OSC_INDEX,DEMOD_INDEX,0,1)

xData_l1, yData_l1, fittedParameters_l1, fit_eval_l1 = read_and_fit(data,sample_node,1)

# Plot initial run on coarse sweeps
axs.plot(xData_l1, yData_l1, 'o', markersize=0.5, label='run' + str(i))
axs.plot(xData_l1, fit_eval_l1, c='red')
plt.pause(0.005)

# Define parameter indices
a_inx = 0
c_inx = 1
w_inx = 2

# Initialize the second figure for parameter tracking
fig2, ax2 = plt.subplots(3,1)

# Plot the initial fitted parameters
ax2[0].plot(-1,fittedParameters_l1[a_inx],'o')
ax2[1].plot(-1,fittedParameters_l1[c_inx],'o')
ax2[2].plot(-1,fittedParameters_l1[w_inx],'o')
plt.pause(0.005)

# Initialize w with the initial width value
a = [fittedParameters_l1[a_inx]]
w = [fittedParameters_l1[w_inx]]

def perform_sweep_and_update(bias, i):
    device.auxouts[1].offset(bias)
    data, sample_node = sweep_now(device, 1.6e3, 3.2e3, 150, OSC_INDEX, DEMOD_INDEX, 0,i)
    _, yData, fp, fit_eval = read_and_fit(data, sample_node, 1)
    return fp, fit_eval, yData

#'Nudge' bias in one direction
bias += 0.001 
fp,fit_eval,_ = perform_sweep_and_update(bias, 2)

a.append(fp[a_inx])
w.append(fp[w_inx])

# Plot the first sweep results
ax2[0].plot(0, fp[a_inx], 'o',c='black')
ax2[0].set_ylabel('Amplitude')

ax2[1].plot(0, fp[c_inx], 'o',c='black')
ax2[1].set_ylabel('Central Frequency')

ax2[2].plot(0, fp[w_inx], 'o',c='black')
ax2[2].set_ylabel('Width (Hz)')

plt.pause(0.005)

#Loop parameters
i=1
stable_count = 0
stable_threshold = 5  # Number of consecutive iterations to confirm stability
percent_threshold = 0.005  # 1% threshold for stability check
max_iterations = 50 #Maximum number of iterations before exiting (timeout condition)
direction = 0.001  # Initial direction (positive bias change)

print('Optimising...')

while True:
  
    fp,fit_eval, yData = perform_sweep_and_update(bias, i+2)
    
    current_w = fp[w_inx]
    
    # Append new width value
    current_w = fp[w_inx]
    w.append(current_w)
    a.append(fp[a_inx])
    
    # Plot data
    ax2[0].plot(i, fp[a_inx], 'o',c='black')
    ax2[1].plot(i, fp[c_inx], 'o',c='black')
    ax2[2].plot(i, fp[w_inx], 'o',c='black')
    plt.pause(0.005)
    
    # Check the direction of change in width
    if len(w) > 1:
        previous_w = w[-2]
        if current_w > previous_w:
            direction = -direction  # Change direction if the width increased
    
    # Update bias
    bias += direction

    # Check the stopping condition
    if len(w) >= 6:  # Ensure we have at least 6 values to check the last 5
        last_5_avg = np.mean(w[-6:-1])
        recent_min = min(w[-6:-1])
        
        if abs(w[-1] - recent_min) / recent_min <= percent_threshold:
            stable_count += 1
        else:
            stable_count = 0  # Reset the counter if the condition is not met
        
        if stable_count >= stable_threshold:
            print(f"Stopping condition met at iteration {i}. Current w is within {percent_threshold * 100}% of the recent minimum for {stable_threshold} consecutive iterations.")
            break

    if i == max_iterations:
        print(f"{i} iterations reached, check for errors")
        break
        
    i += 1  # Increment index

a_over_w = np.array(a)/np.array(w)

fig3, ax3 = plt.subplots()

ax3.plot(xData_l1, yData_l1, 'o', markersize=0.5)
ax3.plot(xData_l1, fit_eval_l1, c='red', label='First Run')

ax3.plot(xData_l1, yData, 'o', markersize=0.5)
ax3.plot(xData_l1, fit_eval, c='blue', label='Optimised Run')

ax3.set(ylabel='Amplitude',xlabel='Frequency (Hz)')

ax3.legend()

print('Bias optimised.')
print(f'Width = {round(w[-1])} Hz, CF = {round(fp[1])} Hz')

slope_inc = (a_over_w[-1]/a_over_w[0]-1)*100

print(f'A/W slope increase of {round(slope_inc,1)}% during optimisation')

