import os
import cv2
import numpy as np
from PIL import Image


recognizer = cv2.face.createLBPHFaceRecognizer() #initialize the face recognizer trainer..


data_base_path = "E:\\projects\\humanoid\\face recognition\\faceDataBase"

def GetUserId(path):
    faces = []
    Ids = []
    for (dir_name, root_name, files_name) in os.walk(path):
        files = files_name

    for file in files:
        FaceImage = Image.open(path + "\\" + file) #use .convert('L') method if the stored data base images are not in gray scale..
        FaceNp = np.array(FaceImage, dtype = np.uint8) #opencv reads an image only in numpy array format...
        ID = file.split('.')[1]
        faces.append(FaceNp)
        Ids.append(ID)
        cv2.waitKey(10)
    return Ids, faces

        
IDs, Faces = GetUserId(data_base_path)
recognizer.train(Faces, np.array(list(map(int, IDs))))
recognizer.save("E:\\projects\\humanoid\\face recognition\\trainedData.yml")
cv2.destroyAllWindows()

