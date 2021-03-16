# import nltk
# from nltk.stem.lancaster import LancasterStemmer
# import numpy as np
# import tflearn
# import tensorflow as tf
# from tensorflow.python.framework import ops
# import random
# import json

# stemmer = LancasterStemmer() #stemmer to get stem of a word. ex. 'say' would be stem word of 'saying'.

# with open('intents.json') as file:
#     data = json.load(file)

# print(type(data))
# print(data)

# # gives stemmed, tokenized words list from sentence pattern without words in ignore_words list
# def clean_pattern(pattern, ignore_words):
#     stemmed_pattern = []
#     wrds = nltk.word_tokenize(pattern)
#     for w in wrds:
#         if w not in ignore_words:
#             stemmed_pattern.append(stemmer.stem(w.lower()))
#     return stemmed_pattern

# stemmed_words = []
# tags = []
# ignore_words = ['!', '?', '.']
# corpus = []

# for intent in data['intents']:
#     for pattern in intent['patterns']:
#         stemmed_pattern = clean_pattern(pattern, ignore_words)
#         stemmed_words.extend(stemmed_pattern)
#         corpus.append((stemmed_pattern, intent['tag']))
#     if intent['tag'] not in tags:
#         tags.append(intent['tag'])
        
# # remove duplicates and sort
# stemmed_words = sorted(list(set(stemmed_words)))
# tags = sorted(list(set(tags)))

# print(stemmed_words)
# print(tags)
# print(corpus)

# X = []
# y = []

# for item in corpus:
#     bag = [] #array of 1 and 0. 1 if stemmed word is present stemmed pattern
#     stemmed_pattern = item[0]
#     for w in stemmed_words:
#         if w in stemmed_pattern:
#             bag.append(1)
#         else:
#             bag.append(0)
            
#     tags_row = [] #array of 1 and 0. 1 for current tag and for everything else 0.
#     current_tag = item[1]
#     for tag in tags:
#         if tag == current_tag:
#             tags_row.append(1)
#         else:
#             tags_row.append(0)
    
#     #for each item in corpus, X will be array indicating stemmed words and y array indicating tags
#     X.append(bag)
#     y.append(tags_row) 

# X = np.array(X)
# y = np.array(y)
# print(X)
# print(y)

# # so our aim is to predict the tag for given sentence and give response based on that tag.


# ops.reset_default_graph() #reset graph data

# # neural network's layers
# network = tflearn.input_data(shape= [None, len(X[0])]) #input layer
# network = tflearn.fully_connected(network, 8) #1st hidden layer
# network = tflearn.fully_connected(network, 8) #2nd hidden layer
# network = tflearn.fully_connected(network, len(y[0]), activation= 'softmax') #output layer
# network = tflearn.regression(network)

# # fitting model
# model = tflearn.DNN(network, tensorboard_dir='tflearn_logs') #tensorboard_dir is path to store logs
# model.fit(X, y, n_epoch=600, batch_size=8, show_metric=True, shuffle= True) #n_epoch:no. of times model will see same data
# model.save('chatbot_model.tflearn')
