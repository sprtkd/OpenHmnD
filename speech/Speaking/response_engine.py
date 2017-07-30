# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 00:15:11 2017

@author: user
"""

import speech_recognition as sr
import pyttsx3
import nltk
import numpy as np
# restore all of our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']


# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# load our saved model
model.load('./model.tflearn')

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return(random.choice(i['responses']))

            results.pop(0)

            if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return(random.choice(i['responses']))

def all_ears():
    r = sr.Recognizer()
    print("listening............")
    with sr.Microphone() as src: 
       
        audio = r.listen(src)
    msg = ''
    try:
        msg = r.recognize_google(audio) 
        #print("message-",msg.lower())
    except sr.UnknownValueError:
        msg='gibberish'
    except sr.RequestError as e:
        print("Could not request results from Google STT; {0}".format(e))
    except:
        msg=''
            
    finally:
        return msg.lower()  
              
def main():
    engine=pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    rate=engine.getProperty('rate')
    engine.setProperty('rate', rate-10)
   
    while(1):
      
    
          message=all_ears()
          if (message == 'gibberish'):
              message=''
              reply="Did you say something?"
              print("Alice:-",reply)
              engine.say(reply)
              engine.runAndWait()
          elif (message!=''):
              
              print("You:",message)
              category= [x[0] for x in classify(message)]
              print(category[0])
              reply=response(message)
              print("Alice:-",reply)
              engine.say(reply)
              engine.runAndWait() 
              if (category[0] =='goodbye'):
                break
          else:
              print("Nothing heard")
              
         
      
    #print(message.lower())
    
    

if __name__== '__main__':                      
	main()
    
    
    

    


