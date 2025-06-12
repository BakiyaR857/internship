import json
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn.functional as F
import pickle
import os

nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
   return sentence.lower().split()

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

def preprocess_data(json_path='data/intents.json'):
    with open(json_path, 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Save data
    os.makedirs("model", exist_ok=True)
    with open('model/data.pth', 'wb') as f:
        torch.save({
            "X_train": X_train,
            "y_train": y_train,
            "all_words": all_words,
            "tags": tags
        }, f)

    print("✅ Data preprocessed and saved to model/data.pth")

# Only run if executed directly
if __name__ == "__main__":
    preprocess_data()
