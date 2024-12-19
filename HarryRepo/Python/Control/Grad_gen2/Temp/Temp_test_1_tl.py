import tldevice
import time

try:
    if 't1' not in globals():
        t1 = tldevice.Device('COM5')
except:
    
    
    t1.therm.sensor.type(1)
    
    t1.therm.sensor.coef(0.3850)
    
    t1.therm.sensor.R0(1000)
    
    t1.therm.sensor.T0(22.7)

    t1.therm.pid.setpoint(40)
    
    t1.therm.pid.enable(1)
    
    while True:
        # Call the functions every second
        print(t1.data())          # Get data
        
        # Wait for 1 second before the next iteration
        time.sleep(1)