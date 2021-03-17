import tflearn
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np

stemmer = LancasterStemmer() #stemmer to get stem of a word. ex. 'say' would be stem word of 'saying'.

def define_network(X, y):
	tf.compat.v1.reset_default_graph() #Clears the default graph stack and resets the global default graph
	# neural network's layers
	network = tflearn.input_data(shape= [None, len(X[0])]) #input layer
	network = tflearn.fully_connected(network, 8) #1st hidden layer
	network = tflearn.fully_connected(network, 8) #2nd hidden layer
	network = tflearn.fully_connected(network, len(y[0]), activation= 'softmax') #output layer
	network = tflearn.regression(network)
	model = tflearn.DNN(network, tensorboard_dir='tflearn_logs') #tensorboard_dir is path to store logs
	return model

# gives stemmed, tokenized words list from sentence pattern without words in ignore_words list
def clean_pattern(pattern, ignore_words):
    stemmed_pattern = []
    wrds = nltk.word_tokenize(pattern)
    for w in wrds:
        if w not in ignore_words:
            stemmed_pattern.append(stemmer.stem(w.lower()))
    return stemmed_pattern

# generates a numpy array of 0 & 1 from string sentence of user to fed to model
def bag_of_words(sentence, stemmed_words, ignore_words):
	bag = []
	stemmed_pattern = clean_pattern(sentence, ignore_words)
	for w in stemmed_words:
		if w in stemmed_pattern:
			bag.append(1)
		else:
			bag.append(0)
	return np.array(bag)