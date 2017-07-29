# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 08:14:24 2017

@author: Punyajoy Saha
"""

import csv
import numpy as np
import cv2
import sys 
import os
import datetime
def add_class(name):
    label=[]
    index=0
    with open('label.csv', newline='') as csvfile:
       spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
       count=0
       for row in spamreader:
         label.append(str(row))
         count=count+1
    name_modi='[\''+name+'\']'
    if name_modi not in label:
        file=open('data/label.pbtxt','a')
        file.write('item {\n  id: '+str(count)+'\n  name: \''+name+'\'\n}\n\n')
        with open('label.csv', 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([name])
        index=len(label)
    else:
        index=label.index(name_modi)
    return index    

    
         
         
#f=open('label.csv','a') 
#rect = (0,0,0,0)
startPoint = False
endPoint = False
folder_created='training/'
if not os.path.exists(folder_created):
         os.makedirs(folder_created)
def on_mouse(event,x,y,flags,params):

    global rect,startPoint,endPoint

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if startPoint == True and endPoint == True:
            startPoint = False
            endPoint = False
            rect = (0, 0, 0, 0)

        if startPoint == False:
            rect = (x, y, 0, 0)
            startPoint = True
        elif endPoint == False:
            rect = (rect[0], rect[1], x, y)
            endPoint = True

cap = cv2.VideoCapture(0)
waitTime = 50
rect=[0,0,0,0]
#Reading the first frame
(grabbed, frame) = cap.read()
breadth=5
height=5

while(cap.isOpened()):
    
    (grabbed, frame) = cap.read()
    rect[0]=int(frame.shape[1]/2)-breadth
    rect[1]=int(frame.shape[0]/2)-height
    rect[2]=int(frame.shape[1]/2)+breadth
    rect[3]=int(frame.shape[0]/2)+height
    cv2.namedWindow('frame')
#    cv2.setMouseCallback('frame', on_mouse)    
    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
    
    
    cv2.imshow('frame',frame)
    key = cv2.waitKey(waitTime) 

    if key == 27:
        break
    elif key == ord("q"):            #q is inc breadth
        if breadth+5<frame.shape[1]:
            breadth=breadth+5
    elif key == ord("w"):            #w is dec breadth
        if breadth-5>0:
            breadth=breadth-5
    elif key == ord("e"):           #e is inc height
        if height+5<frame.shape[0]:
            height=height+5
    elif key == ord("r"):           #r is dec height
        if height-5>0:
            height=height-5
    elif key == ord("s"):
        now = datetime.datetime.now()
        filename=folder_created+str(now.year)+str(now.month)+str(now.day)+str(now.hour)+str(now.minute)+str(now.second)+'.jpg'
        class_ob=input('give the class')
        index=add_class(class_ob)
        cv2.imwrite(filename,frame)
        height=frame.shape[0]
        width=frame.shape[1]
        xmin=rect[0]
        ymin=rect[1]
        xmax=rect[2]
        ymax=rect[3]
        with open('names.csv', 'w') as csvfile:
            fieldnames = ['file_name', 'class_ob','class_index','height','width','xmin','xmax','ymin','ymax']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'file_name': filename, 'class_ob': class_ob,'class_index':index,})
            

    
        
        
        
cap.release()
cv2.destroyAllWindows()


   
#add_class('cat')
#add_class('cat')
#add_class('dog')
#add_class('person')
#f.close()    