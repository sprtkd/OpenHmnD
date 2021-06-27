# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 16:53:01 2017

@author: SANDIPAN
"""
t_path = ['', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\python35.zip', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\DLLs', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\lib', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\lib\\site-packages', 'C:\\Users\\SANDIPAN\\Anaconda3\\envs\\tensorflow\\lib\\site-packages\\setuptools-27.2.0-py3.5.egg']
import sys
sys.path = t_path+sys.path
import pickle
import numpy as np
max_length = pickle.load(open("max_length.p","rb"))
features=np.zeros((1,max_length),dtype='int32')

import tensorflow as tf
import speechtest
from speechtest.speech_test import *
import responseEngine
from responseEngine import *
import chatbot
from chatbot.text_to_idea import *







with tf.Session() as sess:
         
         saver = tf.train.Saver()
         
         saver.restore(sess, "F:/Projects II/speech/saved_model/classifier1/model.ckpt")
         
         stmt = input('Your statement: ')
         features_1 = make_featuresets(stmt)
         features[0,:]=np.array(features_1)
         prediction=sess.run(output,feed_dict={x:features})
         s=np.argmax(prediction)
         if s==0:
             print('command')
             response_engine.botActs(stmt)
         else:
             print('non_command')
             botSpeaks(stmt)