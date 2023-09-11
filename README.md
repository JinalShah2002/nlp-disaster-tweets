# Disaster Tweets
This repository stores my solution for the [Natural Language Processing with Disaster Tweets Kaggle Challenge](https://www.kaggle.com/competitions/nlp-getting-started). 

## Problem
Twitter has become a grand space for communication. With everyone having a smartphone, it becomes very easy to communicate information to a wide audience. Specifically, Twitter allows for a person to communicate an emergency they're observing in real-time. This allows for world organizations such as disaster relief organizations to make informed decisions and allocate resources such that they can help people. Because of this, these organizations want to have a means to monitor Twitter using software-based solutions. However, it is not always clear whether a tweet is referring to a disaster or not. For humans, it becomes easy to see whether a tweet is referring to a disaster or not given context and maybe some visual aid. But, this task is quite difficult for machines. 

## Goal
The goal of this project is to deliver a machine learning solution that can predict which tweets are about real disasters and which ones aren't. 

## System Design (Overview)
Here is what the solution will look like visually:
![image](https://github.com/JinalShah2002/nlp-disaster-tweets/assets/28205508/e7e72939-ef50-4a7d-913f-85e9e11a7559)
As you can see, essentially, the customer sends a tweet to the model, and the model returns a 0, which means the tweet isn't about a disaster, or 1, which means the tweet is about a disaster.

## Data
For this challenge, I am already given the data. The data can be found in the competition site under the Data tab. The training data is made up of TK tweets. The dataset also has 3 other feature columns: ID, Keyword, and Location. ID corresponds to the unique tweet ID, Keyword corresponds to a particular keyword from the tweet ,and location corresponds to the location the tweet was sent from. It is important to note that most examples have the Keyword and Location features empty. Furthermore, to clarify (if it wasn't obvious already) all features are strings (non-numerical or categorical). 

## Solution
It is evident that the solution will have to be some kind of classifier as this problem is a classification problem. I will need to translate the text into some vector/matrix based format such that the models can ingest the data. I can utilize classical machine learning approaches as well as deep learning approaches. 

The Kaggle challenge also has some test data for me to submit. Once the model is trained ,and I am satisfied with the performance, I can feed the test data into the model and make predictions to submit. 

### Evaluation
It is critical to figure out what my evaluation metric is going to be. For this problem, the Kaggle challenge page utilizes the [F1-Score](https://en.wikipedia.org/wiki/F-score). Hence, I will choose my model based on the F1-Score. 

Note, I will also keep track of the following "secondary" evaluation metrics: Precision, Recall, and Accuracy.


