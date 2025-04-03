import serial

###############################################################################
#Write current values to the Pustelny Power Supply (PCS)

def setbias(vals):
    
    def runit():
        PCS = serial.Serial(port='COM9', 
                            baudrate=115200,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            bytesize=serial.EIGHTBITS,
                            timeout=0)
    
        str1 ='!set;1;4mA;{}'.format(vals[0])
        str2 ='!set;2;4mA;{}'.format(vals[1])
    
        PCS.write(bytes(str1+'\r','utf-8'))
        PCS.write(bytes(str2+'\r','utf-8'))
    
        PCS.close()
        
    try:
        runit()
        
    except: #If port is already open, it is closed first
        
        PCS.close()
        runit()
         
    return

###############################################################################
#Execute

vals = [314,-300] #C1,C2 bias current in micro-Amps5

setbias(vals)

#Roughly 103 per 500Hz
