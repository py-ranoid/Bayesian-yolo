#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from PIL import Image


class listDataset(Dataset):

    def __init__(self, root, shuffle=True, transform=None, target_transform=None):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath, label = self.lines[index].split()
        img = Image.open(imgpath).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        label = int(label)
        
        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)
