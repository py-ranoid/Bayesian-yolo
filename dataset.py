#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
from image import *

class listDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       #if self.train:
       #    print('init seen to %d' % (self.seen))

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train and index % 64== 0:
            if self.seen < 4000*64*4:
               width = 13*32
               self.shape = (width, width)
            elif self.seen < 8000*64*4:
               width = (random.randint(0,3) + 13)*32
               self.shape = (width, width)
            elif self.seen < 12000*64*4:
               width = (random.randint(0,5) + 12)*32
               self.shape = (width, width)
            elif self.seen < 16000*64*4:
               width = (random.randint(0,7) + 11)*32
               self.shape = (width, width)
            else: # self.seen < 20000*64*4:
               width = (random.randint(0,9) + 10)*32
               self.shape = (width, width)
        
        jitter = 0.2
        hue = 0.1
        saturation = 1.5 
        exposure = 1.5

        img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
        label = torch.from_numpy(label)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + 4
        return (img, label)
