import os
from face.face_emotion import emotion_detect

#some self init
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #to avoid tensorflow bullsh!t warnings



#init part
emotion_detect.init()



#the main loop, childs are still not created
while(1):
   filepath = input("Enter filepath to detect emotion:  ")
   emotion_detect.predict_mood(filepath)
