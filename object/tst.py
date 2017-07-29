# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:29:01 2017

@author: Punyajoy Saha
"""

#!/usr/bin/env python
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *










#
#import speech2
#
## say() speaks out loud.
#speech2.say("I am speaking out loud.")
#
## input() waits for user input.  The prompt text is optional.
##spoken_text = speech2.input("Say something, user!")
##print ("You said: %s" % spoken_text)
#
## You can limit user input to a set of phrases.
#spoken_text = speech2.input("Are you there, user?", ["Yes", "No", "Shut up, computer."])
#print ("You said: %s" % spoken_text)
#
## If you don't want to wait for input, you can use listenfor() to run a callback
## every time a specific phrase is heard.  Meanwhile your program can move on to other tasks.
#def L1callback(phrase, listener):
#  print ("Heard the phrase: %s" % phrase)
## listenfor() returns a Listener object with islistening() and stoplistening() methods.
#listener1 = speech2.listenfor(["any of", "these will", "match"], L1callback)
#       
## You can listen for multiple things at once, doing different things for each.
#def L2callback(phrase, listener):
#  print ("Another phrase: %s" % phrase)
#listener2 = speech2.listenfor(["good morning Michael"], L2callback)
#
## If you don't have a specific set of phrases in mind, listenforanything() will
## run a callback every time anything is heard that doesn't match another Listener.
#def L3callback(phrase, listener):
#  speech2.say(phrase) # repeat it back
#  if phrase == "stop now please":
#    # The listener returned by listenfor() and listenforanything()
#    # is also passed to the callback.
#    listener.stoplistening()
#listener3 = speech2.listenforanything(L3callback)
#
## All callbacks get automatically executed on a single separate thread.
## Meanwhile, you can just do whatever with your program, or sleep.
## As long as your main program is running code, Listeners will keep listening.
#
#import time
#while listener3.islistening(): # till "stop now please" is heard
#  time.sleep(1)
#
#assert speech2.islistening() # to at least one thing
#print ("Dictation is now stopped.  listeners 1 and 2 are still going.")
#
#listener1.stoplistening()
#print ("Now only listener 2 is going")
#
## Listen with listener2 for a while more, then turn it off.
#time.sleep(30)
#
#speech2.stoplistening() # stop all remaining listeners
#assert not speech2.islistening()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#














#print ('item {\n'+'  name: "my"'+'  name: "my"')
#print ('  name: "my"')
#print ('  name: "my"')
#print ('}')
#
#import cv2
#import numpy as np
#import random
#roi=cv2.imread('img2.png')
#hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#cv2.imwrite('imghsv.png',hsv_roi)
#x_max=hsv_roi.shape[0]
#y_max=hsv_roi.shape[1]
#y_10=int(y_max/20)
#x_10=int(x_max/20)
#a=np.zeros((5,3),dtype='uint8')
#x=random.sample(range(int(x_max/2-20),int(x_max/2+20)),5)
#y=random.sample(range(int(y_max/2-10),int(y_max/2+10)),5)
#
#for i in range(0,a.shape[0]):
#    a[i,0]=hsv_roi[int(x[i]),int(y[i]),0]
#    a[i,1]=hsv_roi[int(x[i]),int(y[i]),1]
#    a[i,2]=hsv_roi[int(x[i]),int(y[i]),2]
#max_0=np.max(a[:,0])
#max_1=np.max(a[:,1])
#max_2=np.max(a[:,2])
#min_0=np.min(a[:,0])
#min_1=np.min(a[:,1])
#min_2=np.min(a[:,2])
#
#
#mask = cv2.inRange(hsv_roi, np.array((min_0, min_1,min_2)), np.array((max_0,max_1,max_2)))
#cv2.imwrite('mask.png',mask)
#roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ) 
