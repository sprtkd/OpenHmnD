"""Facial Emotion Detector
prerequisites - Keras, numpy
By: Suprotik Dey
Special Thanks to: Sethu Iyer
"""

#import
from keras.applications import VGG16,imagenet_utils
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import Model

