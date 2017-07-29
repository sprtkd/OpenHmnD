# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:30:22 2017

@author: Punyajoy Saha
"""

from __future__ import print_function
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2
import speech_recognition as sr
import six
import time
import speech
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import random
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range
#from PIL import Image
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TO_CKPT='E:/computerscience/my_projects/humanoid/tensor_flow/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
IMAGE_SIZE = (12, 8)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


sess=tf.Session(graph=detection_graph)