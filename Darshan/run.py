#This is the actual code where all modules will be called and controlled
#Created by Suprotik Dey
#Robodarshan
#-------------------------------------

"""
The module will work in the following way:
==========================================
The image_analysis and speech_analysis models
will run parallel as threads and communicate.
It will raise interrupt flag if service is required
And pause if pause flag is raised.

To be serviced, the decide_service will decide what to do 
if a certain issue is present and will procedurally call the following modules:
speech_systhesis
hardware_mapper
---------------------------------------------------

"""

#importing modules
import os
import threading
from image import image_analysis
from speech import speech_analysis
from speech import speech_synthesis
from ai import decide_service
from hardware import hardware_mapper


#flags and globals
isImageServiceable = 0
isSpeechServiceable = 0
debug_mode = 0
image_isLive = 1
speech_isLive = 1
speech_return
image_return
check_list = []


#some self init
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to avoid tensorflow bullsh!t warnings
flagLock = threading.lock()#for locking

#init all modules
image_analysis.init(debug_mode)
speech_analysis.init(debug_mode)
speech_synthesis.init(debug_mode)
decide_service.init(debug_mode)
hardware_mapper.init(debug_mode)



def image_service():
    global image_return, isImageServiceable
    while True:
	   flagLock.acquire()
	   live = image_isLive
	   flagLock.release()
	   if live:
	       isServiceable, image_module_return = image_analysis()
		   flagLock.acquire()
	       isImageServiceable = isServiceable
		   image_return = image_module_return
	       flagLock.release()
	   

	
def speech_service():
    global speech_return, isSpeechServiceable
    while True:
	   flagLock.acquire()
	   live = speech_isLive
	   flagLock.release()
	   if live:
	       isServiceable, speech_module_return = speech_analysis()
		   flagLock.acquire()
	       isSpeechServiceable = isServiceable
		   speech_return = speech_module_return
	       flagLock.release()
	   
		   

def decision_maker():
    global image_isLive, speech_isLive, isSpeechServiceable,isImageServiceable,service_acknowledge
	while True:
	    flagLock.acquire()
	    img_tobeServiced = isImageServiceable
	    speech_tobeServiced = isSpeechServiceable
	    flagLock.release()
		#speech high priority
		if speech_tobeServiced:
		    flagLock.acquire()
		    isSpeechServiceable = 0
	        returnval = speech_return
	        flagLock.release()
			IsSpchSynth,IsHdwCallable,spchSynthRet,HdwRet, img_living, sp_living=decide_service.speech(returnval)
			flagLock.acquire()
			image_isLive = img_living
			speech_isLive = sp_living
		    flagLock.release()
			
        elif img_tobeServiced:
		    flagLock.acquire()
		    isImageServiceable = 0
	        returnval = image_return
	        flagLock.release()
		    IsSpchSynth,IsHdwCallable,spchSynthRet,HdwRet, img_living, sp_living=decide_service.img(returnval)
			flagLock.acquire()
			image_isLive = img_living
			speech_isLive = sp_living
		    flagLock.release()
		
		#procedural calls
		if IsSpchSynth:
		    speech_synthesis(spchSynthRet)
		if IsHdwCallable:
		    hardware_mapper.send(HdwRet)
	    
		flagLock.acquire()
		#lets make the threads live again if they are paused..
	    image_isLive = 1
		speech_isLive = 1
		flagLock.release()
	


#create the actual threads
t1 = threading.thread(target=image_service)
t1.daemon = True
t2 = threading.thread(target=speech_service)
t2.daemon = True

print("\nInit Complete!")

decision_maker()
t1.start()
t2.start()



# To show user the humanoid is ready..
print("The humanoid is ready.. You can start!")
