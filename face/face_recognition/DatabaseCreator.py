#FACE RECOGNITION USING USING OPENCV LIBRARY
#THIS WILL REQUIRE THE HAAR CASCADE CLASSIFIER
#HUMANOID PROJECT IIEST, SHIBPUR
#REFERENCE OF THIS CODE: https://www.youtube.com/watch?v=6gWS2CdtZrs
#EDITED BY VIVEK SHARMA....
#you can download the haar cascade features from the given link: https://github.com/opencv/opencv/tree/master/data/haarcascades
#                                                                 http://docs.opencv.org/2.4.13.2/modules/objdetect/doc/cascade_classification.html
#or you can train your own cascade features...

import cv2
import numpy as np
import sqlite3



#FUNCTION TO STORE THE DATA OF THE CORRESPONDING FACES
def UpdateDataBase(id, name):
    #people is a sqlite created file
    connect_sq = sqlite3.connect("FACEDB.db")#open the sqlite created data base
    cmd = "SELECT * FROM people WHERE ID = " + str(id) #command which search for the given ID in the data base
    cursor = connect_sq.execute(cmd)#execute the command, cursor will give all the data row by row
    is_record_exist = 0
    for row in cursor:#check if the ID is already present
        is_row_exist = 1

    if( is_record_exist == 1):#update the data base
        cmd = "UPDATE people SET NAME" + str(name) + "WHERE ID" + str(id)
    else:#insert the data
        cmd = "INSERT INTO people(ID, NAME) VALUES(" + str(id) + "," + str(name) + ")"
    connect_sq.execute(cmd)
    connect_sq.commit()
    connect_sq.close()




#initializing the cascade feature to read the file...
face_cascade = cv2.CascadeClassifier("E:\\projects\\humanoid\\face detection\\haarcascade_frontalface_default.xml")

camera_feed = cv2.VideoCapture(1)


identifier = input("Enter the Face ID: ")#store the ID of the current face, so that we can identify the face later..
name = str(input("Enter the Face NAME: "))
UpdateDataBase(identifier, name)
count = 0;

while True:
    ret, img = camera_feed.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert the image to gray

    #detectMultiScale method returns the array of the rectangle of the detected face...
    face = face_cascade.detectMultiScale(gray, 1.3, 5) #the values 1.3 and 5 are the default value for the function...

    #extracting the cordinate of the rectangle of the face present in the image...
    for (x, y, w, h) in face:
        count = count + 1
        cv2.imwrite("E:\\projects\\humanoid\\face recognition\\faceDataBase\\face" + str(count) + "." + str(identifier) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv2.waitKey(100)#wait for 50ms after storing each picture...
        
    cv2.imshow("FACES", img)
    cv2.waitKey(1)
    if( count >= 25 ): #store only 25 samples...
        break;

camera_feed.release()
cv2.destroyAllWindows()
