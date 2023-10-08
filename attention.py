"""

@author: Jinal Shah

This file will contain
all code for building 
a general attention model

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    # Constructor
    def __init__(self,vocab_size=1002,embedding_dim=50,hidden_size=40,num_layers=1):
        """
        constructor

        Constructor builds the initial model.
        
        inputs:
        vocab_size: an int specifying the vocabulary size
        embedding_dim: an int specifying the embedding dimension
        hidden_size: an int specifying how big the hidden state should be (output)
        num_layers: an int specifying the number of layers
        max_sentence_length: an int for the maximum length of a sentence
        
        outputs:
        None
        """
        super().__init__()

        # Defining the layers
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=1001)
        self.lstm = nn.LSTM(embedding_dim,hidden_size,num_layers,batch_first=True)
        self.attention_one = nn.Linear(hidden_size,hidden_size)
        self.attention_output = nn.Linear(hidden_size,1)
        self.output_one = nn.Linear(hidden_size,20)
        self.final_output_layer = nn.Linear(20,2)

    
    # Method to define the model
    def forward(self,x):
        """
        forward

        A function to define how the input 
        transforms into the output
        
        inputs:
        x - the input of sentence already padded

        outputs:
        outputs - the output of the model
        """
        # Passing input through the embedding layer
        embeds = self.embedding(x)

        # Passing the embedding through the LSTM
        output_encoder, (_, _) = self.lstm(embeds)

        # Passing the outputs through the attention mechanism
        attention_output = torch.reshape(nn.functional.softmax(self.attention_output(self.attention_one(output_encoder)),dim=1),(-1,1,57))
        context_vector = torch.bmm(attention_output,output_encoder)
        context_vector = context_vector.squeeze(axis=1)
        
        # Putting the context vector through the final ANN
        output = F.relu(self.output_one(context_vector))
        final_output = F.sigmoid(self.final_output_layer(output))

        # returning the output
        return final_output

# Main Method
if __name__ == '__main__':
    test_sentence = [1001] * 57
    test_sentence[0] = 1000
    test_sentence[1] = 381
    test_sentence[2] = 455
    test_sentence[3] = 415
    test_sentence[4] = 301

    test_sentence = np.array(test_sentence)
    test_sentence = np.expand_dims(test_sentence,axis=0)

    # print(test_sentence.shape)

    # Creating the model
    attention_clf = Attention()
    prediction = attention_clf(torch.from_numpy(test_sentence))
    print(prediction)



