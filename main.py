import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy 
import tflearn
import tensorflow
import random
import json

stemmer = LancasterStemmer()

with open('intents.json') as file:
	data = json.load(file)

print(type(data))
print(data)