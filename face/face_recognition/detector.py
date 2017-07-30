#HAAR CASCADE FACE DETECTION
#HUMANOID PROJECT IIEST, SHIBPUR
#REFERENCE OF THIS CODE: https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
#EDITED BY VIVEK SHARMA....
#you can download the haar cascade features from the given link: https://github.com/opencv/opencv/tree/master/data/haarcascades
#                                                                 http://docs.opencv.org/2.4.13.2/modules/objdetect/doc/cascade_classification.html
#or you can train your own cascade features...

import cv2
import numpy as np
import sqlite3


#FUNCTION TO GET THE DATA OF THE ASKED ID FROM THE DATABASE
def GetProfile(id):
    connect_sql = sqlite3.connect("FaceDB.db")
    cmd = "SELECT * FROM people WHERE ID = " + str(id)
    cursor = connect_sql.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    connect_sql.close()
    return profile


#initializing the cascade feature to read the file...
face_cascade = cv2.CascadeClassifier("E:\\projects\\humanoid\\face detection\\haarcascade_frontalface_default.xml")

camera_feed = cv2.VideoCapture(1)

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("E:\\projects\\humanoid\\face recognition\\trainedData.yml")
id = 0

while True:
    ret, img = camera_feed.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert the image to gray

    #detectMultiScale method returns the array of the rectangle of the detected face...
    face = face_cascade.detectMultiScale(gray, 1.3, 5) #the values 1.3 and 5 are the default value for the function...

    #extracting the cordinate of the rectangle of the face present in the image...
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        profile = GetProfile(id)
        cv2.putText(img, "ID: " + str(profile[0]), (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(img, "NAME: " + str(profile[1]), (x,y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)


    cv2.imshow("FACE DETECTION", img)
    r = cv2.waitKey(30) & 0xff #waiting for esc key
    if r == 27:
        break

camera_feed.release()
cv2.destroyAllWindows()
