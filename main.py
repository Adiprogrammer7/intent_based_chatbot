import numpy as np
import pickle
import json
import random
import pyttsx3
from utils import clean_pattern, define_network, bag_of_words

with open('saved_variables.pickle', 'rb') as file:
    stemmed_words, tags, ignore_words, X, y = pickle.load(file) 

with open('intents.json') as file:
    data = json.load(file)

engine = pyttsx3.init()
model = define_network(X, y)
model.load("chatbot_model.tflearn")


# to handle previous context and give advantage to results of that context
def context_func(context, user_input):
	model_input = [bag_of_words(user_input, stemmed_words, ignore_words)]
	results = model.predict(model_input)[0]
	for intent in data['intents']:
		if 'context_filter' in intent:
			if intent['context_filter'] == context:
				# looping through tags and their indices
				for tg_index, tg in enumerate(tags):
					if tg == intent['tag']:
						results[tg_index] += 0.5 
	return results

def chat():
	probability_threshold = 0.4 
	context = ""
	print("Welcome to the intents based chatbot. You can start chatting! (enter 'q' to quit)")
	while True:
		user_input = input("You: ")
		user_input = user_input.lower()
		if user_input == 'q':
			break
		# if context from previous response is there, results of that context gets advantage
		if context:
			results = context_func(context, user_input)
		else:
			model_input = [bag_of_words(user_input, stemmed_words, ignore_words)] #as model is trained on 2d array
			results = model.predict(model_input)[0] #gives array of probabilities for all tags
		for i, j in zip(tags, results):
			print(i, j)
		context = "" #reset the context
		result_index = np.argmax(results) #to get index of max probability
		#to filter out predictions below threshold
		if results[result_index] > probability_threshold: 
			tag = tags[result_index] #tag associated with user_input according to model predicition
			for intent in data['intents']:
				if intent['tag'] == tag:
					response = random.choice(intent['responses'])
					# check if context is set for current intent
					if 'context_set' in intent:
						context = intent['context_set']
					break
		else:
			response = "I didn't understand! Maybe try rephrasing..."
		print("Chatbot: {}".format(response))
		engine.say(response)
		engine.runAndWait()

chat()

# TODO: retrain and test main.py