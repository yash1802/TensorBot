import nltk

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import random
import json


def get_training_data():
    with open('intents.json') as json_data:
        intents = json.load(json_data)['intents']
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    for intent in intents:
        for pattern in intent['patterns']:
            # Tokenize Each words
            w = nltk.word_tokenize(pattern)
            # Add to word list
            words.extend(w)
            # Add to documents in corpus
            documents.append((w, intent['tag']))

            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)

    training = []
    output = []
    output_empty = [0] * len(classes)

    for doc in documents:
        #  Bag o words
        bag = []
        # Tokenized words for pattern
        pattern_words = doc[0]
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

        # Bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    return train_x, train_y, words, classes


if __name__ == "__main__":
    get_training_data()
