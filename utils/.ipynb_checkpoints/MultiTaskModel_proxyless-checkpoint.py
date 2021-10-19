import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    """
    Creates a MTL model with the encoder from "model_backbone" 
    """
    def __init__(self, model_backbone):
        super(MultiTaskModel,self).__init__()
        self.encoder = model_backbone       #fastai function that creates an encoder given an architecture
        self.fc1 = nn.Linear(in_features=1432, out_features=2, bias=True)    #fastai function that creates a head
        self.fc2 = nn.Linear(in_features=1432, out_features=90, bias=True)
        self.fc3 = nn.Linear(in_features=1432, out_features=7, bias=True)

    def forward(self,x):

        x = self.encoder(x)
        gender = self.fc1(x)
        age = self.fc2(x)
        emotions = self.fc3(x)

        return [age, gender, emotions]