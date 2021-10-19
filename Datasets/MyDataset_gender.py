import torch
import pandas
from torchvision.datasets.folder import default_loader
import cv2
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

#custom class for gender images uploading
class MyDataset_gender(torch.utils.data.Dataset):
    def __init__(self,root,phase,classes,transform = None,target_transform=None,loader = default_loader):
        imgs = []
        subdatasets = os.listdir(root)
        for one in subdatasets:
            a_file = open(join(root,one,'annotation_'+str(phase)+'.txt'), "r")
            for line in a_file:
                c = line.split(',')
                imgs.append(tuple([join(root,one,phase,c[0]),int(c[1])]))
            a_file.close()
        self.classes = classes
        self.imgs = imgs
        self.phase = phase
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
 
    def __len__(self):
        return len(self.imgs)