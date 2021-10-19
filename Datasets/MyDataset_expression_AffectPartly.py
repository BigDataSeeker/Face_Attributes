import torch
import pandas
from torchvision.datasets.folder import default_loader
import cv2
import numpy as np
import random
import torchvision
from torchvision import datasets, models, transforms

class MyDataset_expression_AffectPartly(torch.utils.data.Dataset):
    def __init__(self,path_to_annotation,phase,dict_classes,transform = None,loader = default_loader):
        annotation = open(path_to_annotation+str(phase)+'.txt','r')
        self.annotations = annotation.readlines()
        self.dict_classes = dict_classes
        self.loader = loader
        self.transform = transform
        random.shuffle(self.annotations)
        if len(self.annotations) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
    def __getitem__(self, index):
        path, target = self.annotations[index].split(';')
        img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, self.dict_classes[target.replace('\n','')]
 
    def __len__(self):
        return len(self.annotations)