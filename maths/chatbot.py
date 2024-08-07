# Import necessary libraries
import io
import random
import string  # to process python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
import os

# Set the NLTK data directory
nltk.data.path.append(os.path.join(os.getcwd(), 'punkt'))

from nltk.tokenize import PunktSentenceTokenizer

# Verify the exact path to 'english.pickle'
punkt_dir = os.path.join(os.getcwd(), 'punkt')
punkt_path = os.path.join(punkt_dir, 'english.pickle')
print(f"Looking for punkt tokenizer at: {punkt_path}")

# Print directory contents
print("Directory contents of 'punkt':")
if os.path.exists(punkt_dir):
    for file in os.listdir(punkt_dir):
        print(file)
else:
    print(f"Directory not found: {punkt_dir}")
    exit(1)

if not os.path.isfile(punkt_path):
    print(f"File not found: {punkt_path}")
    exit(1)

with open(punkt_path, 'rb') as f:
    punkt_tokenizer = PunktSentenceTokenizer(f)

from nltk.stem import WordNetLemmatizer

# Reading the corpus. For our chatbot, we will use a file with information on
# Miranda House as our corpus.
f = open('mh.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()  # converts to lowercase

# Tokenisation
sent_tokens = punkt_tokenizer.tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Preprocessing. We shall now define a function called LemTokens which will take as input the tokens and return normalized tokens.
lemmer = nltk.stem.WordNetLemmatizer()
# WordNet is a semantic dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword matching. If the user inputs a greeting, the bot returns a greeting response.
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

# We will feed the lines that we want our bot to say while starting and ending a conversation depending upon user’s input.
flag = True
print("MHBot: My name is Miranda. I will answer your queries about Miranda House. If you want to exit, type Bye!")
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye' and user_response != 'bye!':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("MHBot: You are welcome. Bye! Take care.")
        else:
            if greeting(user_response) is not None:
                print("MHBot: " + greeting(user_response))
            else:
                print("MHBot: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("MHBot: Bye! Take care.")
