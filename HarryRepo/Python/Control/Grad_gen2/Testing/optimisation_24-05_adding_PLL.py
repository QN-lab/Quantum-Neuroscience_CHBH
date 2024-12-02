# # Log sequence recorded on 2024/04/24 13:15:56

###############################################################################
###############################################################################
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

T_START = time.time()

plt.close('all')

session = Session("localhost")
device = session.connect_device("DEV3994")

OUT_CHANNEL = 0         #Output channel: Sig out 1
OUT_MIXER_CHANNEL = 1   # UHFLI: 3, HF2LI: 6, MFLI: 1
IN_CHANNEL = 1          #0:
CURR_INDEX = 0
DEMOD_INDEX = 0         #Demodulator Index
OSC_INDEX = 0           #Oscillator isndex (I think we only have 1 oscillator)

#C1,C2 bias current in micro-Amps5
b1_fix = 550
b2_start = 350

f_sweep_i = 1.2e3
f_sweep_f = 3.0e3

current_voltage = 31.2 #Set Laser
PLL_INDEX = 0  # PLL index, typically 0 if you have only one PLL

if 'device' in locals():
    device.pids[PLL_INDEX].enable(False)
    
    
#SET NODES
with device.set_transaction():
  
    device.currins[CURR_INDEX].on(True)
    device.currins[CURR_INDEX].autorange()
    device.currins[CURR_INDEX].float(0)                           #Floating ground    OFF
    device.currins[CURR_INDEX].scaling(1)                         #Input scaling      1V/1V
    device.currins[CURR_INDEX].range(0.000010)   
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
    
    
#Write current values to the Pustelny Power Supply (PPS)

def setbias(vals):
    
    def runit():
        PPS = serial.Serial(port='COM9', 
                            baudrate=115200,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            bytesize=serial.EIGHTBITS,
                            timeout=0)
    
        str1 ='!set;1;4mA;{}'.format(vals[0])
        str2 ='!set;2;4mA;{}'.format(vals[1])
    
        PPS.write(bytes(str1+'\r','utf-8'))
        PPS.write(bytes(str2+'\r','utf-8'))
    
        PPS.close()
        
    try:
        runit()
        
    except: #If port is already open, close and reopen
        
        PPS.close()
        runit()
         
    return


vals = [b1_fix,b2_start]
setbias(vals)

host = '172.29.169.14'

def set_scan_offset(value):
    # Value in Volts
    with DLCpro(NetworkConnection(host)) as dlcpro:
        dlcpro.laser1.scan.offset.set(value)
        print(f'Scan Offset adjusted: {value}V')


set_scan_offset(current_voltage)

with DLCpro(NetworkConnection(host)) as dlcpro:
    current_voltage = dlcpro.laser1.scan.offset.get()
    print(f"Current scan offset value: {current_voltage}V")

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
    sweeper.bandwidthcontrol(0.01) #Changes bandwidth
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
        [maxY/1.5, maxY*2],         # parameter bounds for A
        [maxY/1.5, maxY*2],         # parameter bounds for A1
        [minX, maxX],               # parameter bounds for c0
        [minX, maxX],               # parameter bounds for c1
        [10, 150],                  # parameter bounds for w0 (HWHM)
        [10, 150],                  # parameter bounds for w1 
        [-0.005, 0.005],            # parameter bounds for slope
        [maxY/-0.005, maxY/0.005]   # parameter bounds for offset
    ]
    

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
    in_phase_data = node_samples[0][0]["y"]
    
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
    
    return xData, yData, fittedParameters, fit_eval, in_phase_data
###############################################################################
###############################################################################
#%%Coarse sweep

start_b2 = 350
stop_b2 = 600
num = 5

print(f'Performing Coarse Bias search. Number of sweeps: {num}')

biases = np.linspace(start_b2,stop_b2,num=num)

perr_list = np.zeros((8,len(biases)))
param_list = np.zeros((8,len(biases)))

r_squared_list = list()

fit_eval_list = list()

fig, axs = plt.subplots()

for i, bias in enumerate(biases): 
    
    setbias([b1_fix,bias])
    
    data, sample_node = sweep_now(device,f_sweep_i,f_sweep_f,150,OSC_INDEX,DEMOD_INDEX,0,i+1)

    xData, yData, fittedParameters, fit_eval, _ = read_and_fit(data,sample_node,2)

    param_list[:,i] = fittedParameters

    residuals = yData - Lorentzian_double(xData,*fittedParameters)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yData-np.mean(yData))**2)
    r_squared = 1-(ss_res/ss_tot)
    r_squared_list.append(r_squared)
    
    axs.plot(xData,yData,'o',markersize = 0.5,label='run' + str(i))
    axs.plot(xData,fit_eval,'--',c='blue')
    plt.pause(0.005)
    
    fit_eval_list.append(fit_eval)
    

r_squared_list = np.array(r_squared_list)

fit_eval_list = np.stack(fit_eval_list, axis=1)

###############################################################################
###############################################################################
#%% 
#Swap to single-Lorentzian and optimise amplitude/width around this region.

print('Fitting Single Lorentzian')

# opt_inx = np.argmax(r_squared_list) #Find best run by R squared

opt_inx = np.argmax(np.max(fit_eval_list, axis=0))

bias = biases[opt_inx]              #Re-adjust bias to this bias value

setbias([b1_fix,bias])

data, sample_node = sweep_now(device,f_sweep_i,f_sweep_f,150,OSC_INDEX,DEMOD_INDEX,0,1)

xData_l1, yData_l1, fittedParameters_l1, fit_eval_l1,_ = read_and_fit(data,sample_node,1)

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
    
    setbias([b1_fix,bias])
    
    data, sample_node = sweep_now(device, f_sweep_i, f_sweep_f, 150, OSC_INDEX, DEMOD_INDEX, 0,i)
    _, yData, fp, fit_eval,_ = read_and_fit(data, sample_node, 1)
    return fp, fit_eval, yData

#'Nudge' bias in one direction
bias += 3
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
ax2[2].set_xlabel('Iteration Number')

plt.pause(0.005)

#Loop parameters
i=1
stable_count = 0
stable_threshold = 5  # Number of consecutive iterations to confirm stability
percent_threshold = 0.01  # 1% threshold for stability check
max_iterations = 50 #Maximum number of iterations before exiting (timeout condition)
direction = 2  # Initial direction (positive bias change)

print('Optimising bias...')

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


print('Bias optimised.')

slope_inc = (a_over_w[-1]/a_over_w[0]-1)*100

print(f'A/W slope increase of {round(slope_inc,1)}% during optimisation')

###############################################################################
###############################################################################
#%% Optimise Laser
print('Reading Laser..')

def perform_sweep_and_laser(voltage, j):
    set_scan_offset(voltage)
    data, sample_node = sweep_now(device, f_sweep_i, f_sweep_f, 150, OSC_INDEX, DEMOD_INDEX, 0, j)
    xData, quad_data, fp, fit_eval, inp_data = read_and_fit(data, sample_node, 1)
    return fp, fit_eval, quad_data ,xData, inp_data

fig4, ax4 = plt.subplots(4, 1)
ax4[0].set_ylabel('Amplitude')
ax4[1].set_ylabel('Central Frequency')
ax4[2].set_ylabel('Width (Hz)')
ax4[3].set_ylabel('A/W')
ax4[3].set_xlabel('Iteration Number')

fp, fit_eval, yData,_, _ = perform_sweep_and_laser(current_voltage, 0)

current_voltage += 0.1

awl = list([fp[a_inx] / fp[w_inx]])

# Loop parameters
j = 1
stable_count = 0
stable_threshold = 5  # Number of consecutive iterations to confirm stability
percent_threshold = 0.05  # 5% threshold for stability check
max_iterations = 20  # Maximum number of iterations before exiting (timeout condition)
direction = 0.1  # Initial direction (positive bias change)

print('Optimising Laser...')

awl = list()
w = list()
a = list()

while True:
    fp, fit_eval_l, yData_l, xData_l, inp_data_l = perform_sweep_and_laser(current_voltage, j)

    w.append(fp[w_inx])
    a.append(fp[a_inx])

    # Append new width value
    current_aw = a[-1] / w[-1]
    awl.append(current_aw)

    # Plot data
    ax4[0].plot(j, fp[a_inx], 'o', c='black')
    ax4[1].plot(j, fp[c_inx], 'o', c='black')
    ax4[2].plot(j, fp[w_inx], 'o', c='black')
    ax4[3].plot(j, current_aw, 'o', c='black')
    
    plt.pause(0.005)

    # Check the direction of change in awl
    if len(awl) > 1:
        previous_aw = awl[-2]
        if current_aw < previous_aw:
            direction = -direction  # Change direction if awl decreased
    
    # Update bias
    current_voltage += direction

    # Check the stopping condition
    if len(awl) >= 6:  # Ensure we have at least 6 values to check the last 5
        last_5_avg = np.mean(awl[-6:-1])
        recent_max = max(awl[-6:-1])

        if abs(awl[-1] - recent_max) / recent_max <= percent_threshold:
            stable_count += 1
        else:
            stable_count = 0  # Reset the counter if the condition is not met
        
        print(str(stable_count))
        if stable_count >= stable_threshold:
            print(f"Stopping condition met at iteration {j}. Current awl is within {percent_threshold * 100}% of the recent maximum for {stable_threshold} consecutive iterations.")
            break

    if j == max_iterations:
        print(f"{j} iterations reached, check for errors")
        break

    j += 1  # Increment index


fig3, ax3 = plt.subplots(2,1)

ax3[0].plot(xData_l1, yData_l1, 'o', markersize=0.5)
ax3[0].plot(xData_l1, fit_eval_l1, c='red', label='First Run')

ax3[0].plot(xData_l1, yData, 'o', markersize=0.5)
ax3[0].plot(xData_l1, fit_eval, c='blue', label='Bias-Optimised Run')


ax3[0].plot(xData_l1, yData_l, 'o', markersize=0.5)
ax3[0].plot(xData_l1, fit_eval_l, c='green', label='Laser-Optimised Run')

ax3[0].set(ylabel='Amplitude',xlabel='Frequency (Hz)')

ax3[0].legend()
plt.pause(0.005)

print(f'Width = {round(w[-1])} Hz, CF = {round(fp[1])} Hz')

#%% 
#PLL locking, might want to do anther sweep here. 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

amplitude = a[-1]
width = w[-1]
central_f = fp[1]

#Calculate Phase: 
freq = xData_l
quadrature = yData_l
in_phase = inp_data_l

phi = np.angle(quadrature + 1j * in_phase,deg=True)

new_domain = np.linspace(freq[0],freq[-1],5000)

interped_phi = np.interp(new_domain,freq,phi)


ax3[1].plot(freq,phi,'o',c='red',label='Real Phi Data')
ax3[1].plot(new_domain,interped_phi,'.',c='k',label='Interpolated Phi',alpha=0.1)
ax3[1].set(xlabel='Frequency',ylabel='Phase',autoscale_on=1)

PLL_setpoint = interped_phi[find_nearest(new_domain,central_f)]

ax3[1].axvline(x=central_f,c='blue',label='Central Frequency')
ax3[1].axhline(y=PLL_setpoint,c='green',label=f'PLL Setpoint = {round(PLL_setpoint,2)}')
ax3[1].legend(loc=1)

print(f'Setpoint: {PLL_setpoint}')
    
with device.set_transaction():
    device.pids[PLL_INDEX].mode(1)
    device.pids[PLL_INDEX].input(3)
    device.pids[PLL_INDEX].inputchannel(0)
    device.pids[PLL_INDEX].output(2)
    device.pids[PLL_INDEX].outputchannel(0)
    device.pids[PLL_INDEX].phaseunwrap(0)
    device.pids[PLL_INDEX].setpoint(PLL_setpoint)
    device.pids[PLL_INDEX].center(central_f)
    device.pids[PLL_INDEX].limitlower(-200)
    device.pids[PLL_INDEX].limitupper(200)
    
    device.pids[PLL_INDEX].demod.timeconstant(0.82e-3) #100Hz
    device.pids[PLL_INDEX].demod.order(3)
    device.pids[PLL_INDEX].demod.harmonic(1)
    
    device.pids[PLL_INDEX].p(-50e-3)
    device.pids[PLL_INDEX].i(-100e-6)
    device.pids[PLL_INDEX].d(0)
    device.pids[PLL_INDEX].dlimittimeconstant(0)
 

if abs(PLL_setpoint) > 10:
    answer = input('PLL setpoint > 10 deg, continue?')  # Added the missing closing parenthesis
    if answer.lower() in ["y", "yes"]:
        # Enable the PLL
        device.pids[PLL_INDEX].enable(True)  # Correctly indented this line
    elif answer.lower() in ["n","no"]:
        print('PLL lock cancelled due to high phase lock point')
        sys.exit()
else:
    
    device.pids[PLL_INDEX].enable(True)  # Correctly indented this line
    print('PLL Enabled...')


T_END = time.time()
T_ELAPSED = round((T_END-T_START)/60,1)

print(f'Time Elapsed: {T_ELAPSED} mins')



