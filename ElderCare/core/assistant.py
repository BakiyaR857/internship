import torch
import random
import json
import sys
import os

# Add parent directory to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.rnn_model import RNNModel
from model.preprocess import bag_of_words, tokenize
from core.speak import speak  # For voice output

# Load intents
with open('data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained model data
FILE = "model/rnn_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Create the model and load the state
model = RNNModel(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Function to get bot response
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).long().unsqueeze(0)

    with torch.no_grad():
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent["responses"]), tag
        return "I'm not sure I understand. Can you try again?", "unknown"

# Main interactive loop
if __name__ == "__main__":
    print("🤖 ElderCare Assistant: Hello! How can I assist you today? (type 'quit' to exit)")
    speak("Hello! How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "stop"]:
            print("🤖 Assistant: Goodbye! Take care.")
            speak("Goodbye! Take care.")
            break

        response, tag = get_response(user_input)
        print(f"🤖 Assistant: {response}")
        speak(response)

