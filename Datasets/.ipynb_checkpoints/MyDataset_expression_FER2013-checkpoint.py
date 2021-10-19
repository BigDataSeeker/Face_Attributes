import torch
import pandas
from torchvision.datasets.folder import default_loader
import cv2
import numpy as np
import random
import torchvision
from torchvision import datasets, models, transforms

class MyDataset_expression_FER2013(torch.utils.data.Dataset):
    def __init__(self,path_to_csv,phase,dict_classes,transforms = None):
        csv_dataset = open(path_to_csv,'r')
        self.csv_dataset = [ line for line in csv.reader(csv_dataset)]
        garbage = self.csv_dataset.pop(0)
        self.dict_classes = dict_classes
        self.phase = phase
        self.transforms = transforms
        split = int(0.2*len(self.csv_dataset))
        if phase == 'train':
            self.csv_dataset = self.csv_dataset[split:]
        if phase == 'val':
            self.csv_dataset = self.csv_dataset[:split]
        

        if len(self.csv_dataset) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
    def __getitem__(self, index):
        target_subjective, csv_line_values = self.csv_dataset[index]
        pixel_values = [int(value) for value in csv_line_values.split()]
        pixel_values = np.array(pixel_values)
        pixel_values = np.reshape(pixel_values,(48,48))
        pixel_values_array = [pixel_values,pixel_values,pixel_values]
        rgb_like_img = np.stack(pixel_values_array,axis = -1)
        img = PIL.Image.fromarray(rgb_like_img, 'RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.dict_classes[int(target_subjective)]
 
    def __len__(self):
        return len(self.csv_dataset)