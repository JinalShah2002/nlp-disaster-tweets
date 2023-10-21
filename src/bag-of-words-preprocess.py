"""

@author: Jinal Shah

This file will preprocess the raw 
data and produce a bag of words 
representation of each example.

Bag-of-words is the simpliest representation of
text in NLP. It essentially is a vector of the count
of each word in the vocabulary in a given text.

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from preprocessing import Preprocessing

# Importing the data
raw_train = pd.read_csv('data/train.csv')
raw_test = pd.read_csv('data/test.csv')

training, testing = train_test_split(raw_train,test_size=0.2,random_state=42,shuffle=True,stratify=raw_train['target'])

# Filling in the missing keywords with None
training['keyword'] = training['keyword'].fillna('None')
testing['keyword'] = testing['keyword'].fillna('None')

# Creating a file for the keyword probabilities in the training data
keyword_counts_total = dict(training['keyword'].value_counts())
keyword_count_one = dict(training.loc[training['target'] == 1, 'keyword'].value_counts())
keyword_probs = {}

# Getting the percentages for every keyword and writing them to the file
with open('keyword-probs.json','w') as file:
    for keyword in keyword_counts_total.keys():
       keyword_probs[keyword] = 0

       if keyword in keyword_count_one.keys():
           keyword_probs[keyword] = keyword_count_one[keyword] / keyword_counts_total[keyword]

    file.write(json.dumps(keyword_probs)) 

# Getting the preprocessed sentences
preprocessor = Preprocessing()
cleaned_train_tweets = preprocessor.preprocess_data(training.drop('target',axis=1))

# Creating a dictionary with the counts of each word in each tweet
word_counts = []
for sentence in cleaned_train_tweets:
    sentence_tokens = {}
    for word in sentence:
        if word in sentence_tokens.keys():
            temp = sentence_tokens[word] + 1
            sentence_tokens[word] = temp
        else:
            sentence_tokens[word] = 1
    word_counts.append(sentence_tokens)

# Creating the bag of words vector for each example
bag_of_words = []
training.reset_index(drop=True,inplace=True)

with open('mappers/word2index.json') as file:
    vocabulary = json.load(file)
    
    for index in range(0,len(word_counts)):
        vector = {}

        for word in vocabulary.keys():
            if word in word_counts[index].keys():
                vector[word] = word_counts[index][word]
            else:
                vector[word] = 0
        
        vector['ID'] = training.loc[index,'id']
        vector['Keyword'] = training.loc[index,'keyword']
        vector['Target'] = training.loc[index,'target']

        bag_of_words.append(vector)

# Converting bag_of_words to a dataframe
bag_of_words_train = pd.DataFrame(bag_of_words)
bag_of_words_train.to_csv('data/bag_of_words_training.csv',index=False)

# Performing the same transformations on the testing
testing.reset_index(drop=True,inplace=True)
preprocessed_test_tweets = preprocessor.preprocess_data(testing.drop('target',axis=1))

# Creating a dictionary with the counts of each word in each tweet
word_counts = []
for sentence in preprocessed_test_tweets:
    sentence_tokens = {}
    for word in sentence:
        if word in sentence_tokens.keys():
            temp = sentence_tokens[word] + 1
            sentence_tokens[word] = temp
        else:
            sentence_tokens[word] = 1
    word_counts.append(sentence_tokens)

# Creating the bag of words vector for each example
bag_of_words = []

with open('mappers/word2index.json') as file:
    vocabulary = json.load(file)
    
    for index in range(0,len(word_counts)):
        vector = {}

        for word in vocabulary.keys():
            if word in word_counts[index].keys():
                vector[word] = word_counts[index][word]
            else:
                vector[word] = 0
        
        vector['ID'] = testing.loc[index,'id']
        vector['Keyword'] = testing.loc[index,'keyword']
        vector['Target'] = testing.loc[index,'target']

        bag_of_words.append(vector)

bag_of_words_test = pd.DataFrame(bag_of_words)
bag_of_words_test.to_csv('data/bag_of_words_testing.csv',index=False)