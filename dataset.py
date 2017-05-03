#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths


class listDataset(Dataset):

    def __init__(self, root, shuffle=True, transform=None, target_transform=None):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.size = (544,544)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt')
        label = torch.zeros(50*5)

        if os.path.getsize(labpath):
            tmp = torch.from_numpy(np.loadtxt(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            else:
                label[0:tsz] = tmp

        img = Image.open(imgpath).convert('RGB')
        img = img.resize(self.size)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)
