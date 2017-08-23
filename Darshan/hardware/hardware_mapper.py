#hardware mapper and caller for OpenHmnD 1.0

import serial


# connect to serial monitor
serial_obj=0

def init():
    global serial_obj
	serial_obj = serial.Serial('COM3',9600)
	serial_obj.close()
	serial_obj.open() #open channel

def send(send_var):
    serial_obj.write(send_var)

def read():
    return serial_obj.read()
	
