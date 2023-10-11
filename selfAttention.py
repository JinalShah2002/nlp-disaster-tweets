"""

@author: Jinal Shah

This file is dedicated to building the self-Attention 
model 

"""
# Importing libraries 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    # Constructor
    def __init__(self,vocab_size=1002,embedding_dim=40) -> None:
        """
        constructor

        The constructor builds the model.

        inputs:
        vocab_size: the size of the vocabulary
        embedding_dim: the dimension of embedding

        outputs:
        - None
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,embedding_dim)

        # Getting the weights for converting the embedding to query, value, and key
        self.query = nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.key = nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.value = nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.scaling_factor = np.sqrt(embedding_dim)

        # Getting the output using an ANN stacked on top of the self-attention model
        self.output1 = nn.Linear(embedding_dim,50,bias=True)
        self.output = nn.Linear(50,2,bias=True)
    
    # A method for running the model
    def forward(self,x):
        """
        forward

        A method to take the input x and run it through the model

        inputs:
        - x: the matrix of inputs

        outputs:
        - output: the output
        """
        # Putting the input through the self-attention mechanism
        embed = self.embedding(x)
        query_matrix = self.query(embed)
        key_matrix = self.key(embed)
        value_matrix = self.value(embed)
        input_matrix = torch.bmm(query_matrix,torch.transpose(key_matrix,1,2)) / self.scaling_factor
        attention_output = torch.bmm(F.softmax(input_matrix,dim=-1),value_matrix)

        # Converting the attention output to a context vector
        # Can be done by averaging the vectors as per the following paper:
        # Self-Attention: A Better Building Block for Sentiment Analysis Neural Network Classifiers
        context_vector = torch.mean(attention_output,dim=1)

        # Putting the output through the ANN to get the output
        output_one = F.relu(self.output1(context_vector))
        final_output = F.sigmoid(self.output(output_one))

        # returning the output
        return final_output

