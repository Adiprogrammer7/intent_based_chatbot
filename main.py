import numpy as np
import pickle
import json
import random
from utils import clean_pattern, define_network, bag_of_words

with open('saved_variables.pickle', 'rb') as file:
    stemmed_words, tags, ignore_words, X, y = pickle.load(file) 

with open('intents.json') as file:
    data = json.load(file)

def chat():
	model = define_network(X, y)
	model.load("chatbot_model.tflearn")
	print("Welcome to the intents based chatbot. You can start chatting! (enter 'q' to quit)")
	while True:
		user_input = input("You: ")
		user_input = user_input.lower()
		if user_input == 'q':
			break
		model_input = [bag_of_words(user_input, stemmed_words, ignore_words)] #as model is trained on 2d array
		results = model.predict(model_input) #gives array of probabilities for all tags
		result_index = np.argmax(results) #to get index of max probability 
		tag = tags[result_index] #tag associated with user_input according to model predicition
		for intent in data['intents']:
			if intent['tag'] == tag:
				response = random.choice(intent['responses'])
				break
		print("Chatbot: {}".format(response))

chat()