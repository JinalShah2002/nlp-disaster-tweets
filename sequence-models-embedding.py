"""

@author: Jinal Shah

This file will be for running the same sequence models
but they all have an embedding layer now.

"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import SimpleRNN, Dense, GRU, LSTM, Embedding
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessing
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Importing the data
raw_train = pd.read_csv('data/train.csv')

# Splitting the data into training, validation, and testing
training, testing = train_test_split(raw_train,test_size=0.2,random_state=42,shuffle=True,stratify=raw_train['target'])
validation, testing = train_test_split(testing,test_size=0.2,random_state=42,shuffle=True,stratify=testing['target'])
training.reset_index(drop=True,inplace=True)
validation.reset_index(drop=True,inplace=True)
testing.reset_index(drop=True,inplace=True)

# Splitting data into X & Y
train_x = training.drop(['target'],axis=1)
train_y = training['target'].values
valid_x = validation.drop(['target'],axis=1)
valid_y = validation['target'].values

# Getting the preprocessed text
preprocessor = Preprocessing()
preprocessed_train_x = preprocessor.preprocess_data(train_x)
preprocessed_valid_x = preprocessor.preprocess_data(valid_x)

# Transforming data to (number of examples, 57, 1000)
training_X = []
valid_X = []

# Opening the vocabulary and marking 1 to indicate the word
with open('mappers/word2index.json') as file:
    vocabulary = json.load(file)

    # Iterating through the training
    for sentence_index in range(0,len(preprocessed_train_x)):
        sentence_convert = [1001] * 57
        for word_index in range(0,len(preprocessed_train_x[sentence_index])):
            # If the word is in the vocab, get the index
            word = preprocessed_train_x[sentence_index][word_index]
            if word in vocabulary.keys():
                sentence_convert[word_index] = vocabulary[word]
            else:
                sentence_convert[word_index] = 1000
        
        training_X.append(sentence_convert)
    
    # Iterating through the validation
    for sentence_index in range(0,len(preprocessed_valid_x)):
        sentence_convert = [1001] * 57
        for word_index in range(0,len(preprocessed_valid_x[sentence_index])):
            # If the word is in the vocab, get the index
            word = preprocessed_valid_x[sentence_index][word_index]
            if word in vocabulary.keys():
                sentence_convert[word_index] = vocabulary[word]
            else:
                sentence_convert[word_index] = 1000

        valid_X.append(sentence_convert)

training_X = np.array(training_X)
valid_X = np.array(valid_X)

# Creating an array for the model metrics
training_metrics = []
validation_metrics = []

# Creating a function that returns the metrics
def get_metrics(truth,predictions):
    f1 = f1_score(truth,predictions)
    precision = precision_score(truth,predictions)
    recall = recall_score(truth,predictions)
    accuracy = accuracy_score(truth,predictions)
    return f1, precision, recall, accuracy

# Model 1: Recurrent Neural Network
print('Starting Recurrent Neural Network...')
rnn_clf = Sequential()
rnn_clf.add(Embedding(input_dim=1002,output_dim=50,input_length=training_X.shape[1]))
rnn_clf.add(SimpleRNN(units=50,activation='tanh',return_sequences=False))
rnn_clf.add(Dense(1,activation='sigmoid',bias_initializer='ones'))
loss_function = 'binary_crossentropy'
optimizer = keras.optimizers.Adam(learning_rate=0.001)
rnn_clf.compile(optimizer,loss_function)
history = rnn_clf.fit(x=training_X,y=train_y,batch_size=32,epochs=30)

training_predictions = rnn_clf.predict(training_X,batch_size=32)
validation_predictions = rnn_clf.predict(valid_X,batch_size=32)
train_metrics = get_metrics(train_y,np.rint(training_predictions))
valid_metrics = get_metrics(valid_y,np.rint(validation_predictions))
train_metrics_df = {'Name':'Recurrent Neural Network','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'Recurrent Neural Network','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished Recurrent Neural Network Model')
print()

# Plotting the Loss Function
# plt.title('Recurrent Neural Network Loss')
# plt.plot(history.history['loss'])
# plt.show()

# Model 2: Gated Recurrent Unit
print('Starting Gated Recurrent Unit...')
gru_clf = Sequential()
gru_clf.add(Embedding(input_dim=1002,output_dim=50,input_length=training_X.shape[1]))
gru_clf.add(GRU(units=50,activation='tanh',recurrent_activation='sigmoid',bias_initializer='ones',return_sequences=False))
gru_clf.add(Dense(1,activation='sigmoid',bias_initializer='ones'))
loss_function = 'binary_crossentropy'
optimizer = keras.optimizers.Adam(learning_rate=0.001)
gru_clf.compile(optimizer,loss_function)
history = gru_clf.fit(x=training_X,y=train_y,batch_size=32,epochs=30)
training_predictions = gru_clf.predict(training_X,batch_size=32)
validation_predictions = gru_clf.predict(valid_X,batch_size=32)
train_metrics = get_metrics(train_y,np.rint(training_predictions))
valid_metrics = get_metrics(valid_y,np.rint(validation_predictions))
train_metrics_df = {'Name':'Gated Recurrent Unit','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'Gated Recurrent Unit','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished Gated Recurrent Unit Model')
print()

# Model 2: Long-Short Term Memory
print('Starting Long-Short Term Memory...')
lstm_clf = Sequential()
lstm_clf.add(Embedding(input_dim=1002,output_dim=50,input_length=training_X.shape[1]))
lstm_clf.add(LSTM(units=50,activation='tanh',recurrent_activation='sigmoid',return_sequences=False))
lstm_clf.add(Dense(1,activation='sigmoid',bias_initializer='ones'))
loss_function = 'binary_crossentropy'
optimizer = keras.optimizers.Adam(learning_rate=0.001)
lstm_clf.compile(optimizer,loss_function)
history = lstm_clf.fit(x=training_X,y=train_y,batch_size=32,epochs=30)
training_predictions = lstm_clf.predict(training_X,batch_size=32)
validation_predictions = lstm_clf.predict(valid_X,batch_size=32)
train_metrics = get_metrics(train_y,np.rint(training_predictions))
valid_metrics = get_metrics(valid_y,np.rint(validation_predictions))
train_metrics_df = {'Name':'Long-Short Term Memory','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'Long-Short Term Memory','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished Long-Short Term Memory Model')
print()

training_metrics_df = pd.DataFrame(training_metrics)
validation_metrics_df = pd.DataFrame(validation_metrics)

print('Training Metrics:')
print(training_metrics_df)
print()
print('Validation Metrics:') 
print(validation_metrics_df)

# Saving the metrics
training_metrics_df.to_csv('model-performances/training-sequence-embedding.csv',index=False)
validation_metrics_df.to_csv('model-performances/validation-sequence-embedding.csv',index=False)