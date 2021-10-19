import torch
import torch.nn as nn

class MultiTaskModel_grouped_age_head(nn.Module):
    """
    Creates a MTL model with the encoder from "model_backbone" 
    """
    def __init__(self, model):
        super(MultiTaskModel_grouped_age_head,self).__init__()
        self.encoder = model     
        self.idx_tensor = torch.from_numpy(np.array([idx for idx in range(31)])).to(device)
        self.age_group_head = nn.Linear(in_features=1400, out_features=31, bias=True)
        self.Softmax = nn.Softmax(1)
    def forward(self,x):

        age,gender,emotions = self.encoder(x)

        grouped_age = self.age_group_head(age)
        regression_age = torch.sum(self.Softmax(grouped_age) * self.idx_tensor, axis=1)*3
  

        return [gender, (grouped_age,regression_age),  emotions]