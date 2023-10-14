"""


@author: Jinal Shah

This file will train and evaluate 
the BERT model on this problem set.

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessing
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from bert import BERT

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

# Tokenizing the data
# BERT has a specific vocabulary and way of handling words not in vocab
# so for optimal performance, we should use the tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

# Getting the tokenized data
training_X = []
valid_X = []
for sentence_index in range(0,len(preprocessed_train_x)):
    training_X.append(tokenizer.encode(preprocessed_train_x[sentence_index],padding='max_length',truncation=True,max_length=57))

for sentence_index in range(0,len(preprocessed_valid_x)):
    valid_X.append(tokenizer.encode(preprocessed_valid_x[sentence_index],padding='max_length',truncation=True,max_length=57))

train_X = torch.from_numpy(np.array(training_X))
valid_X = torch.from_numpy(np.array(valid_X))
train_y = torch.from_numpy(train_y)
valid_y = torch.from_numpy(valid_y)

training_dataset = TensorDataset(train_X,train_y)
validation_dataset = TensorDataset(valid_X,valid_y)

# Storing the training and validation data in DataLoaders
training_loader = DataLoader(training_dataset,batch_size=32,shuffle=True)
validation_loader = DataLoader(validation_dataset,batch_size=32,shuffle=True)

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

# A function for the training loop
def training_loop(model,loss_fn,optimizer,training_data):
    size = len(training_data.dataset)
    model.train() # Setting the model to training mode

    # Iterating through the batches
    for batch , (X,y) in enumerate(training_data):
        # Compute the predictions
        pred = model(X)

        # Calculate the loss
        loss = loss_fn(pred,y)

        # Calculate the derivatives (backpropagation)
        loss.backward()

        # Take a step with the optimizer
        optimizer.step()

        # Reset the gradients
        optimizer.zero_grad()

        # Printing out the progress for every 20 batches
        if batch % 20 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f'loss :{loss} {round(current/size,2)*100}% Complete')

# Training the model
model = BERT()
epochs = 5
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_function = torch.nn.CrossEntropyLoss()
history = []

# Training the model
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')
    training_loop(model,loss_function,optimizer,training_loader)
    training_pred = model(train_X)
    loss = loss_function(training_pred,train_y)
    history.append(loss.item())
    print()

training_predictions = np.argmax(model(train_X).detach().numpy(),axis=1)
validation_predictions = np.argmax(model(valid_X).detach().numpy(),axis=1)
train_metrics = get_metrics(train_y,np.rint(training_predictions))
valid_metrics = get_metrics(valid_y,np.rint(validation_predictions))
train_metrics_df = {'Name':'BERT','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'BERT','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished BERT Model')
print()

training_metrics_df = pd.DataFrame(training_metrics)
validation_metrics_df = pd.DataFrame(validation_metrics)
print('Training Metrics:')
print(training_metrics_df)
print()
print('Validation Metrics:') 
print(validation_metrics_df)

# Saving the metrics
training_metrics_df.to_csv('model-performances/training-BERT.csv',index=False)
validation_metrics_df.to_csv('model-performances/validation-BERT.csv',index=False)

# Plotting the learning curve
plt.title('Training Loss vs. Epochs')
plt.plot(history)
plt.show()