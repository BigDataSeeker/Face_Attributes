import torch
import torch.nn as nn

class MultiTaskModel_baseline(nn.Module):
    """
    Creates a model with the encoder from "model_backbone" 
    """
    def __init__(self, model_backbone):
        super(MultiTaskModel_baseline,self).__init__()
        self.encoder = model_backbone       #fastai function that creates an encoder given an architecture
        self.fc1 = nn.Linear(in_features=512, out_features=2, bias=True)    #fastai function that creates a head
        self.fc2 = nn.Linear(in_features=512, out_features=90, bias=True)
        self.fc3 = nn.Linear(in_features=512, out_features=7, bias=True)
        self.age_group_head = nn.Linear(in_features=90, out_features=31, bias=True)
        self.idx_tensor = torch.from_numpy(np.array([idx for idx in range(31)])).to(device)
        self.Softmax = nn.Softmax(1)
    def forward(self,x):

        x = self.encoder(x)
        gender = self.fc1(x)
        age = self.fc2(x)
        emotions = self.fc3(x)
        grouped_age = self.age_group_head(age)
        regression_age = torch.sum(self.Softmax(grouped_age) * self.idx_tensor, axis=1)*3
        return [gender, (grouped_age,regression_age),  emotions]