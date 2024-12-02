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
BIAS_OFFSET_1 = 0.270

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
    return data, sample_node

    # if save==1:
    #     sweeper.save.save(True)
    #     # Wait until the save is complete. The saving is done asynchronously in the background
    #     # so we need to wait until it is complete. In the case of the sweeper it is important
    #     # to wait for completion before before performing the module read command. The sweeper has
    #     # a special fast read command which could otherwise be executed before the saving has
    #     # started.
    #     sweeper.save.save.wait_for_state_change(True, invert=True, timeout=5)
        
    #     print("SAVED DATA")
        
    #     return data, sample_node
    # else:
        
    #     return data, sample_node

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
def sumOfSquaredError_double(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    return np.sum((yData - Lorentzian_double(xData, *parameterTuple)) ** 2)

def generate_Initial_Parameters_double(xData,yData):
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)
    
    parameterBounds = []
    
    parameterBounds.append([0.0, maxY*2]) # parameter bounds for A
    parameterBounds.append([0.0, maxY*2]) # parameter bounds for A1
    parameterBounds.append([minX, maxX]) # parameter bounds for c0
    parameterBounds.append([minX, maxX]) # parameter bounds for c1
    parameterBounds.append([0.0, 150]) # parameter bounds for w0 (HWHM)
    parameterBounds.append([0.0, 150]) # parameter bounds for w1 
    parameterBounds.append([-0.005, 0.005]) # parameter bounds for slope
    parameterBounds.append([maxY/-0.005, maxY/0.005]) # parameter bounds for offset

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError_double, parameterBounds, seed=3)
    return result.x

#Single Lorentzian
def sumOfSquaredError_single(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    return np.sum((yData - Lorentzian_single(xData, *parameterTuple)) ** 2)

def generate_Initial_Parameters_single(xData,yData):
    # min and max used for bounds
    maxX = max(xData)
    minX = min(xData)
    maxY = max(yData)
    minY = min(yData)
    
    parameterBounds = []
    
    parameterBounds.append([0.0, maxY*2]) # parameter bounds for A
    parameterBounds.append([minX, maxX]) # parameter bounds for x_0
    parameterBounds.append([0.0, 150]) # parameter bounds for w (HWHM)
    parameterBounds.append([-0.005, 0.005]) # parameter bounds for slope
    parameterBounds.append([maxY/-0.005, maxY/0.005]) # parameter bounds for offset

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError_single, parameterBounds, seed=3)
    return result.x

###############################################################################
###############################################################################
#SWEEP CURRENT PARAMETERS

data, sample_node = sweep_now(device,1.9e3,3.5e3,300,OSC_INDEX,DEMOD_INDEX,0)
node_samples = data[sample_node] #extract data from dict

xData = node_samples[0][0]["frequency"] #Frequency values from sweeper
yData = node_samples[0][0]["x"]

initialParameters = generate_Initial_Parameters_double(xData,yData)

# curve fit the test data
fittedParameters, pcov = curve_fit(Lorentzian_double, xData, yData, initialParameters)

perr = np.sqrt(np.diag(pcov))

# #Plot swept data
fig, axs = plot_sweep(node_samples,sample_node)

A0,A1,c0,c1,w0,w1,s,o = fittedParameters
fit = Lorentzian_double(xData,A0,A1,c0,c1,w0,w1,s,o)

axs[0].plot(xData, fit, ls ='--',c='green')

axs[0].plot(xData,(s*xData+o),ls ='--',c='green')
###############################################################################
#Value Report
print('############################################################')
print('Double Fit Parameters')
print('Amplitudes(nA): ' + str(round(A0*1e9,5))+'; '+str(round(A1*1e9,5)))
print('Central freq(Hz): ' + str(round(c0,0))+'; '+str(round(c1,0)))
print('FWHM(Hz): ' + str(round(w0*2,0))+'; '+str(round(w1*2,0)))
print('Slope(nA/Hz): ' + str(round(s*1e9,10)))
print('0Hz intercept(nA): ' + str(round(o*1e9,5)))
print('############################################################')

#%%
#FIRST FITTING

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def pull_fit_range(fData,yData,c,w):

    SearchRange = [c-w/3.5, c+w/3.5] # Goes from left side of one resonance to right side of the other. 

    RangeWidth = SearchRange[1]-SearchRange[0]

    IndexRange = [find_nearest(fData,SearchRange[0]),find_nearest(fData,SearchRange[1])]

    fLin = fData[IndexRange[0]:IndexRange[1]]
    yLin = yData[IndexRange[0]:IndexRange[1]]
    
    return fLin,yLin,RangeWidth,SearchRange

#If one fitted lorentzian is far larger than the other then we have a single lorentzian. 
if A0 > A1*3 or A0*3 < A1 or w0>w1*1.5 or w0*1.5 < w1:
    print('Single Lorentzian detected!')

    initialParameters_l = generate_Initial_Parameters_single(xData,yData)

    # curve fit the test data
    fittedParameters_l, pcov_l = curve_fit(Lorentzian_single, xData, yData, initialParameters_l)

    A0l,c0l,w0l,sl,ol = fittedParameters_l
    fit_l = Lorentzian_single(xData,A0l,c0l,w0l,sl,ol)
    
    axs[0].plot(xData, fit_l,c='red')
    
    axs[0].plot(xData,(sl*xData+ol),c='red')

    print('############################################################')
    print('Single Fit Parameters')
    print('Amplitude(nA): ' + str(round(A0l*1e9,5)))
    print('Central freq(Hz): ' + str(round(c0l,0)))
    print('FWHM(Hz): ' + str(round(w0l*2,0)))
    print('Slope(nA/Hz): ' + str(round(s*1e9,10)))
    print('0Hz intercept(nA): ' + str(round(o*1e9,5)))
    print('############################################################')

    #Linearity analysis
    
    inpData = node_samples[0][0]["y"]
    
    fLin,inpLin,RangeWidth,SearchRange = pull_fit_range(xData,inpData,c0l,w0l)
    
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
    
    #%% 
    #OPTIMISE SYSTEM ITERATIVELY
    
    #Increase bias in one coil and measure resultant slope
    
    adjust = BIAS_OFFSET_1+ 0.001
    
    device.auxouts[1].outputselect(4)
    device.auxouts[1].offset(adjust)
    
    SweepRange = [c0l-2.5*w0l, c0l+2.5*w0l]
   
    data_loc, sample_node_loc = sweep_now(device,int(SweepRange[0]),int(SweepRange[1]),100,OSC_INDEX,DEMOD_INDEX,0)
    
    node_samples_loc = data_loc[sample_node_loc] #extract data from dict
    
    f_loc = node_samples_loc[0][0]["frequency"] 
    x_loc = node_samples_loc[0][0]["x"]
    y_loc = node_samples_loc[0][0]["y"]
    
    fig3, ax3 = plt.subplots(2,1)
    ax3[0].plot(f_loc,x_loc)
    ax3[0].autoscale()
    
    fit_loc, _ = curve_fit(Lorentzian_single, f_loc, x_loc, fittedParameters_l) #fitted params because they shouldn't be far from the last
    
    aAdj,cAdj,wAdj,sAdj,oAdj = fit_loc
    
    fit_adj = Lorentzian_single(f_loc,*fit_loc)
    
    ax3[0].plot(f_loc,fit_adj,c='red')
    
    fAdj,yAdj,RangeWidthAdj,SearchRangeAdj = pull_fit_range(f_loc,y_loc,cAdj,wAdj)
    
    pAdj = np.polyfit(fAdj, yAdj, 1)
    fittedAdj = np.poly1d(pAdj)
    
    ax3[1].plot(f_loc,y_loc,'o',c='orange')
    
    
    slopeAdj = pAdj[0]
    
    ax3[1].plot(fAdj,fittedAdj(fAdj),c='green',label= 's= '+ str(round(slopeAdj,14)))
    ax3[1].legend()
    #LOOP
    #Initial values that will be ovewritten and updated during the loop
    c_iter = cAdj
    w_iter = wAdj
    fit_iter = fit_loc
    
    #Slope matrix, which will be appended to do comparisons
    m_Mat = np.array([slopeLin,slopeAdj])
    
    i = 0
    
    shim = 0.002 #Small adjust to the bias coil
    
    while True:
        
        i += 1
        
        if m_Mat[-1]/m_Mat[-2] > 1: # Moved in the right direction!
            adjust = adjust + shim
        
        if m_Mat[-1]/m_Mat[-2] <= 1: # Moved in the wrong direction!
            adjust = adjust - shim
            
        device.auxouts[1].outputselect(4)
        device.auxouts[1].offset(adjust)
        
        Sr = [c_iter-2.5*w_iter, c_iter+2.5*w_iter]
        data_iter, sample_node_iter = sweep_now(device,int(Sr[0]),int(Sr[1]),80,OSC_INDEX,DEMOD_INDEX,0)
        
        node_samples_iter = data_iter[sample_node_iter] #extract data from dict
        
        f_iter = node_samples_iter[0][0]["frequency"] 
        x_iter = node_samples_iter[0][0]["x"]
        y_iter = node_samples_iter[0][0]["y"]
    
        fit_iter, _ = curve_fit(Lorentzian_single, f_iter, x_iter, fit_iter)
        
        _,c_iter,w_iter,_,_ = fit_iter
        
        f_iter_adj,y_iter_adj,_,_ = pull_fit_range(f_iter,y_iter,c_iter,w_iter)
        
        p_iter = np.polyfit(f_iter_adj, y_iter_adj, 1)
        
        
        m_Mat = np.append(m_Mat,p_iter[0])
        
        if abs(m_Mat[-1])/abs(m_Mat[-1]) < 1.05 and m_Mat[-1]/abs(m_Mat[-1]) > 0.95 : #!!!!!?!?!?!??!?!?!?¬!??!! This is not good
            break
        
        # if abs(m_Mat[-1])/max(abs(m_Mat[:-1])) < 1.05 and abs(m_Mat[-1])/max(abs(m_Mat[:-1])) > 0.95 : #!!!!!?!?!?!??!?!?!?¬!??!! This is not good
        #     break
            
    print(m_Mat)
    print('num of optimisation loops completed: '+ str(i))
    
    fig4, ax4 = plt.subplots(2,1)
    ax4[0].plot(f_iter,x_iter)
    ax4[0].autoscale()
    ax4[0].plot(f_iter,Lorentzian_single(f_iter,*fit_iter),c='red')
    
    lin_fitted_iter = np.poly1d(p_iter)
    
    ax4[1].plot(f_iter,y_iter,'o',c='orange')
    ax4[1].autoscale()
    ax4[1].plot(f_iter_adj,lin_fitted_iter(f_iter_adj),c='green',label= 's= '+ str(round(p_iter[0],14)))
    ax4[1].legend()
    
    
    
    

    
    # i = 1
    
    # Linfit_mat = np.array([slopeLin])
    
    # while True:
        
    #     elif (Linfit_mat[i]/Linfit_mat[i-1]) <= 0.05:
    #         print('Optimised.')
    #         break
        
    #     if Linfit_mat[i] >= Linfit[i-1]
    
    
    #     if 
    
        
    
    # i += 1




