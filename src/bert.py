"""

@author: Jinal Shah

This file will contain the class
for the BERT model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# Class for the BERT Model
class BERT(nn.Module):
    # Constructor
    def __init__(self):
        """
        constructor

        This will contruct the model.

        inputs:
        - None 
        outputs:
        - None
        
        """
        super().__init__()

        # Getting the BERT model and freezing the parameters
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freezing the parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Create the classifier
        self.layer1 = nn.Linear(768,250,bias=True)
        self.layer2 = nn.Linear(250,100,bias=True)
        self.output = nn.Linear(100,2,bias=True)
    
    # Forward Method to get the predictions
    def forward(self,x):
        """
        forward 

        This method will take the input and provide us the output

        inputs:
        - x: the inputs to the model

        outputs:
        - output: the prediction
        """
        output = self.bert(x,output_attentions=False)[1]

        # Passing the output of Bert through the classifer
        output = F.relu(self.layer1(output))
        output = F.relu(self.layer2(output))
        output = F.sigmoid(self.output(output))

        # Returning the output
        return output
