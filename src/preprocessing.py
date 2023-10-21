"""

@author: Jinal Shah

This file will be dedicated to a Preprocessing
class to preprocess the data

"""
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

class Preprocessing():
    # Constructor
    def __init__(self,strip_chars="#-'.;:)([]!?|/*@",replacements={"\n":" ", "\t": " ", "&gt":"", "&lt":"", "&amp":" and "}):
        """
        constructor
        
        Class constructor

        inputs:
        - strip_chars: a string of characters to strip from each word
        - replacements: a dictionary where key is the set of characters and the value is the value to replace
        the characters with.

        outputs:
        - None
        """
        self.strip_chars = strip_chars
        self.replacements = replacements

        # Creating a list for the stop words
        nltk.download('stopwords')
        self.stop_words = stopwords.words('english')
    
    # Creating a function to drop the columns
    def drop_columns(self,data,columns=['location','id','keyword']):
        """
        drop_columns

        A function to drop the listed columns in the data

        inputs:
        - data: data is a Pandas Dataframe containing the data
        - columns: a list of columns to drop

        outputs:
        - The new dataframe with the dropped columns
        """
        return data.drop(columns,axis=1)
        # try:
        #     return data.drop(columns,axis=1)
        # except Exception:
        #     return None
    
    # Creating a function to perform the preprocessing
    def preprocess_data(self,data):
        """
        preprocess_data

        A function to preprocess the data

        inputs:
        - data: a Pandas Dataframe containing the raw data

        outputs:
        - A numpy array containing the preprocessed sentences at each index
        """
        # Dropping the columns
        text_data = self.drop_columns(data)

        # Convert the text data to a numpy array 
        text_data_arr = text_data.to_numpy()
        processed_sentences = []

        # Getting the preprocessed text
        for sentence in text_data_arr:
            # Making the replacements
            preprocessed_sentence = sentence[0]
            for replacements in self.replacements.keys():
                preprocessed_sentence = preprocessed_sentence.replace(replacements,self.replacements[replacements])
            
            # Splitting the sentence by space
            preprocessed_sentence_arr = preprocessed_sentence.split(' ')
            final_preprocessed_sentence = [] # an array for the preprocessed sentence

            for word in preprocessed_sentence_arr:
                processed_word = word.strip(self.strip_chars)
                processed_word = processed_word.lower()
                processed_word = processed_word.encode('ascii','ignore')
                processed_word = processed_word.decode()

                # Conditions to add it to the final preprocessed sentence
                if (processed_word != "") and (processed_word not in self.stop_words) and ("http" not in processed_word) and ("@" not in processed_word):
                    final_preprocessed_sentence.append(processed_word)
                
            processed_sentences.append(final_preprocessed_sentence)
        
        # returning the preprocessed sentences
        return processed_sentences
    
    
