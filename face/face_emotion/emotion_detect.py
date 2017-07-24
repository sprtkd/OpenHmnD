"""Facial Emotion Detector
prerequisites - Keras, numpy
By: Suprotik Dey
Special Thanks to: Sethu Iyer
"""

#import
import keras
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
from keras.applications import VGG16,imagenet_utils
from keras.models import Model

def convert_img_to_vector(img_path):
    image = load_img(img_path,target_size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    image = preprocess(image)
    return image


def get_image_feature(img_path):
    feats = np.transpose(new_model.predict(convert_img_to_vector(img_path)))
    return feats

def predict_mood(img_path, showPercent=True):
    decode_dict={0: 'Angry', 1: 'Disgusted', 2: 'Happy', 3:'Sad', 4:'Scared',5:'Shocked'}
    feats = get_image_feature(img_path)
    feats = feats.reshape(-1,4096)
    probab = model.predict_proba(feats,verbose=0)
    top_2 = probab[0].argpartition(-2)[-2:][::-1]
    percent_high = np.around(100*probab[0][top_2[0]],decimals=2)
    percent_secondhigh = np.around(100*probab[0][top_2[1]],decimals=2)
    if (showPercent):
        print('The person in the image is '+str(percent_high)+' % '+decode_dict[top_2[0]] +' and '+str(percent_secondhigh)+' % '+decode_dict[top_2[1]])
    return (str(percent_high), str(percent_secondhigh))

def init():
    global preprocess, model, new_model
    preprocess = imagenet_utils.preprocess_input
    model = VGG16(weights="imagenet")
    new_model = Model(inputs=model.input,outputs=model.layers[21].output)
    model = load_model('./face/face_emotion/trained_model.h5')


