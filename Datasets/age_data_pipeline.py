import torch
import pandas
from torchvision.datasets.folder import default_loader
import cv2
import numpy as np
import torchvision
import os
from torchvision import datasets, models, transforms

class MyDataset_age(torch.utils.data.Dataset):
    def __init__(self,root,annotation_name,classes,transform = None,loader = default_loader):

        
        self.classes = classes
        self.attribute_frame = pandas.read_csv(os.path.join(root,annotation_name))
        self.root = root
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        img_path , target = os.path.join(self.root, self.attribute_frame.iloc[index, 0]), int(self.attribute_frame.iloc[index, 1])
        group_label = int(target/3)
        #img = self.loader(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image = img)['image']
        
        return img, group_label,target
 
    def __len__(self):
        return len(self.attribute_frame)
    
def dataloader_age(MyDataset,root,item_prob_filename,batch_size,shuffle , num_workers):
    item_prob_file =open(root + "/" + item_prob_filename, "r")
    item_prob_list = item_prob_file.readlines()
    item_prob_list = [float(item_prob_list[i]) for i in range(len(item_prob_list))]
    t = item_prob_list.pop(0)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(item_prob_list, len(item_prob_list))
    return torch.utils.data.DataLoader(MyDataset, batch_size=batch_size, num_workers=num_workers,shuffle=shuffle,sampler=sampler)