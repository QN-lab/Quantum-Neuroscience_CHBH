from pypixxlib._libdpx import (
    DPxOpen, DPxClose, DPxWriteRegCache, DPxUpdateRegCache,
    DPxGetTime, DPxStopDinLog, DPxSetDinLog, DPxStartDinLog,
    DPxGetDinStatus, DPxReadDinLog, DPxEnableDinDebounce
)
from psychopy import core
DPxOpen()
 
logStatus = DPxSetDinLog(0, 1000)
DPxEnableDinDebounce()
DPxStartDinLog()
DPxUpdateRegCache()
startTime = DPxGetTime()

print("Starting to log button events. Press buttons to see their codes.")

finished = False
while not finished:
    DPxUpdateRegCache()
    DPxGetDinStatus(logStatus)
    newData = logStatus['newLogFrames']

 

    if newData > 0:
        log = DPxReadDinLog(logStatus, int(newData))
        for x in log:
            buttonCode = x[1]
            time = round(x[0] - startTime, 2)
            print(f"Button event detected! Code: {buttonCode}, Time: {time}")

 

            if buttonCode == 65519:
                finished = True

 

core.wait(0.25)

DPxStopDinLog()
DPxClose()