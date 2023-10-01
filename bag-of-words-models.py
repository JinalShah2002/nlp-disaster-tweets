"""

@author: Jinal Shah

This file is dedicated to making models
for the Bag of Words text representation

"""
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras import Sequential
import warnings
warnings.filterwarnings('ignore')

# Importing the data
training = pd.read_csv('data/bag_of_words_training.csv')
testing = pd.read_csv('data/bag_of_words_testing.csv')

# Splitting testing data into validation
validation, testing = train_test_split(testing,test_size=0.2,random_state=42,shuffle=True,stratify=testing['Target'])
validation.reset_index(drop=True,inplace=True)

# Splitting data into X & y
train_x = training.drop(['ID','Target'],axis=1)
train_y = training['Target'].values
valid_x = validation.drop(['ID','Target'],axis=1)
valid_y = validation['Target'].values

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

# Base Model: Using the Keywords
training_prediction = []
validation_prediction = []
print('Starting Keywords Base Model...')
with open('mappers/keyword-probs.json') as file:
    keywords_probs = json.load(file)

    for index in range(0,train_x.shape[0]):
        training_prediction.append(round(keywords_probs[train_x.loc[index,'Keyword']]))
    
    for index in range(0,valid_x.shape[0]):
        validation_prediction.append(round(keywords_probs[valid_x.loc[index,'Keyword']]))

train_metrics = get_metrics(train_y,training_prediction)
valid_metrics = get_metrics(valid_y,validation_prediction)
base_model_train = {'Name':'Keyword','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
base_model_valid = {'Name':'Keyword','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(base_model_train)
validation_metrics.append(base_model_valid)
print('Finished Keywords Base Model')
print()

# Model 2: Logistic Regression
print('Starting Logistic Regression Model...')
logReg = LogisticRegression(penalty='none',random_state=42,max_iter=500,n_jobs=-1,solver='sag')
logReg.fit(train_x.drop(['Keyword'],axis=1).values,train_y)
training_predictions = logReg.predict(train_x.drop(['Keyword'],axis=1).values)
validation_predictions = logReg.predict(valid_x.drop(['Keyword'],axis=1).values)
train_metrics = get_metrics(train_y,training_predictions)
valid_metrics = get_metrics(valid_y,validation_predictions)
train_metrics_df = {'Name':'Logistic Regression','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'Logistic Regression','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished Logistic Regression Model')
print()

# Model 3: Decision Trees
print('Starting Decision Trees Model...')
decision_tree = DecisionTreeClassifier(criterion='entropy',random_state=42)
decision_tree.fit(train_x.drop(['Keyword'],axis=1).values,train_y)
training_predictions = decision_tree.predict(train_x.drop(['Keyword'],axis=1).values)
validation_predictions = decision_tree.predict(valid_x.drop(['Keyword'],axis=1).values)
train_metrics = get_metrics(train_y,training_predictions)
valid_metrics = get_metrics(valid_y,validation_predictions)
train_metrics_df = {'Name':'Decision Tree','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'Decision Tree','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished Decision Tree Model')
print()

# Model 4: Random Forests
print('Starting Random Forests Model...')
random_forest = RandomForestClassifier(n_estimators=100,criterion='entropy',bootstrap=True,random_state=42,n_jobs=-1)
random_forest.fit(train_x.drop(['Keyword'],axis=1).values,train_y)
training_predictions = random_forest.predict(train_x.drop(['Keyword'],axis=1).values)
validation_predictions = random_forest.predict(valid_x.drop(['Keyword'],axis=1).values)
train_metrics = get_metrics(train_y,training_predictions)
valid_metrics = get_metrics(valid_y,validation_predictions)
train_metrics_df = {'Name':'Random Forests','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'Random Forests','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished Random Forests Model')
print()

training_metrics_df = pd.DataFrame(training_metrics)
validation_metrics_df = pd.DataFrame(validation_metrics)

# Model 5: AdaBoost Classifier
print('Starting AdaBoost Model...')
adaboost = AdaBoostClassifier(n_estimators=100,base_estimator=DecisionTreeClassifier(),random_state=42)
adaboost.fit(train_x.drop(['Keyword'],axis=1).values,train_y)
training_predictions = adaboost.predict(train_x.drop(['Keyword'],axis=1).values)
validation_predictions = adaboost.predict(valid_x.drop(['Keyword'],axis=1).values)
train_metrics = get_metrics(train_y,training_predictions)
valid_metrics = get_metrics(valid_y,validation_predictions)
train_metrics_df = {'Name':'AdaBoost','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'AdaBoost','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished AdaBoost Model')
print()

# Model 6: Gradient Boosting Classifier
print('Starting Gradient Boosting Model...')
catboost_clf = CatBoostClassifier(iterations=1500,learning_rate=0.07,loss_function='Logloss',random_state=42)
catboost_clf.fit(train_x.drop(['Keyword'],axis=1).values,train_y)
training_predictions = catboost_clf.predict(train_x.drop(['Keyword'],axis=1).values)
validation_predictions = catboost_clf.predict(valid_x.drop(['Keyword'],axis=1).values)
train_metrics = get_metrics(train_y,training_predictions)
valid_metrics = get_metrics(valid_y,validation_predictions)
train_metrics_df = {'Name':'Gradient Boosting','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'Gradient Boosting','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished Gradient Boosting Model')
print()

# Model 7: Artificial Neural Networks
print('Starting Artificial Neural Networks Model...')
ann_clf = Sequential()
ann_clf.add(Dense(25,kernel_initializer='he_normal',bias_initializer='ones',activation='relu',input_shape=(train_x.drop(['Keyword'],axis=1).shape[1],)))
ann_clf.add(Dense(50,kernel_initializer='he_normal',bias_initializer='ones',activation='relu'))
ann_clf.add(Dense(25,kernel_initializer='he_normal',bias_initializer='ones',activation='relu'))
ann_clf.add(Dense(1,kernel_initializer='he_normal',bias_initializer='ones',activation='sigmoid'))

loss_function = 'binary_crossentropy'
optimizer = keras.optimizers.Adam(lr=0.03)
ann_clf.compile(optimizer,loss_function)
history = ann_clf.fit(x=train_x.drop(['Keyword'],axis=1).values,y=train_y,batch_size=32,epochs=500)
training_predictions = ann_clf.predict(train_x.drop(['Keyword'],axis=1).values,batch_size=32)
validation_predictions = ann_clf.predict(valid_x.drop(['Keyword'],axis=1).values,batch_size=32)
train_metrics = get_metrics(train_y,np.rint(training_predictions))
valid_metrics = get_metrics(valid_y,np.rint(validation_predictions))
train_metrics_df = {'Name':'Artificial Neural Network','F1':train_metrics[0],'Precision':train_metrics[1],'Recall':train_metrics[2],'Accuracy':train_metrics[3]}
valid_metrics_df = {'Name':'Artificial Neural Network','F1':valid_metrics[0],'Precision':valid_metrics[1],'Recall':valid_metrics[2],'Accuracy':valid_metrics[3]}

training_metrics.append(train_metrics_df)
validation_metrics.append(valid_metrics_df)
print('Finished Artificial Neural Network Model')
print()

training_metrics_df = pd.DataFrame(training_metrics)
validation_metrics_df = pd.DataFrame(validation_metrics)

print('Training Metrics:')
print(training_metrics_df)
print()
print('Validation Metrics:') 
print(validation_metrics_df)