"""

@author: Jinal Shah

This file is dedicated to building a transformer.
Note, since the problem is sentiment analysis, 
I will only need the encoder of the transformer.

"""
# Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Class for the transformer
class Transformer(nn.Module):
    # Constructor
    def __init__(self,vocab_size=1002,embed_dim=128,heads=4,max_length=57):
        """
        constructor

        This will be the constructor

        inputs:
        - vocab_size: size of the vocabulary
        - embed_dim: the embedding dimension
        - heads: the number of heads in the transformer 
        - max_length of a sentence

        outputs:
        - None
        """
        super().__init__() # Inheriting methods from nn.Module
        self.embed_dim = embed_dim

        # Defining the layers
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,heads,bias=True)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.atten_output = nn.Linear(embed_dim,embed_dim,bias=True)
        self.final_atten_output = nn.Linear(embed_dim,embed_dim,bias=True)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # Creating the positional encoding vector
        self.pos_encode = self.getPositionalEncoding(max_length,embed_dim)

        # Classifier
        self.clf_output_one = nn.Linear(embed_dim,50,bias=True)
        self.output = nn.Linear(50,2,bias=True)
    
    # A function to get the positional encoding
    def getPositionalEncoding(self,max_length,embed_dim):
        """
        getPositionalEncoding

        A function to the get the positional encoding
        for each position.

        inputs:
        - max_length: What is the length of the longest sentence
        - embed_dim: The dimension of the embedding

        outputs:
        - a vector indicating the positional encoding vector
        """
        # Creating a matrix indicating the embedding
        embedding = torch.zeros((max_length,embed_dim))
        dimension_divisor = torch.zeros(embed_dim) # creating a vector to store the divisors

        # Populating the array to contain the dimension divisors
        for i in range(0,embed_dim):
            dimension_divisor[i] = 10000 ** ((2*i)/embed_dim)

        # Iterating through each value and calculating the positional embedding
        for pos in range(embedding.shape[0]):
            for i in range(0,embedding.shape[1],2):
                embedding[pos,i] = torch.sin(pos/dimension_divisor[i])
                embedding[pos,i+1] = torch.cos(pos/dimension_divisor[i+1])
        
        # returning the embedding
        embedding = torch.unsqueeze(embedding,dim=0)
        return embedding
    
    # Function to run the input through the transformer encoder
    def forward(self,x):
        """
        forward

        A method to take the input and produce
        the model output.
        
        inputs:
        - x: a tensor of the input

        outputs:
        - output: the output of the transformer encoder
        """
        model_in = (np.sqrt(self.embed_dim) * self.embedding(x)) + self.pos_encode
        output,_ = self.attention(query=model_in,key=model_in,value=model_in,need_weights=False)
        output = self.layernorm1(model_in + output)

        # Putting the output of multihead attention through the feed-forward
        output_feed = F.relu(self.atten_output(output))
        output_feed = self.final_atten_output(output_feed)
        output = self.layernorm2(output+output_feed)

        # Averaging the 57 embeddings and sending through the classifer
        output = torch.mean(output,dim=1)
        output = F.relu(self.clf_output_one(output))
        output = F.sigmoid(self.output(output))

        return output
        
