# intent_based_chatbot

A intent based chatbot in python with tflearn and tensorflow. It can be trained for a specific purpose and works well within that specific scope. The ```intents.json``` file can
be updated based on purpose and even if statement pattern given from the user varies from the patterns on which model is trained, the model still will give accurate results. The 
model gives probabilities for different tags based on the input and then appropriate response corresponding to that tag is returned.

## To run:
Clone the repo to your local machine:
```
https://github.com/Adiprogrammer7/intent_based_chatbot.git
```
Install dependencies:
```
pip install -r requirements.txt
```
You can run ```main.py``` directly as model is already trained on some sample data. But you can change the scope of chatbot by updating ```intents.json``` file and then training the
model by running ```training_chatbot.ipynb```.

## Preview:
You can see here, the inputs given by the user are not identical to the patterns mentioned in ```intents.json``` file, still model is efficient enough to predict for the trained
scope. 

![preview_screenshot](https://user-images.githubusercontent.com/30752980/112297673-32dcbf00-8cbc-11eb-92a0-760ececf79bb.png)
