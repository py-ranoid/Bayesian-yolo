#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros(max_boxes*5)
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        for i in range(bs.shape[0]):
            bs[i][1] = min(0.999, max(0, bs[i][1] * sx - dx)) 
            bs[i][2] = min(0.999, max(0, bs[i][2] * sy - dy)) 
            bs[i][3] = min(0.999, max(0, bs[i][3] * sx))
            bs[i][4] = min(0.999, max(0, bs[i][4] * sy))
        
            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
        
        tmp = np.reshape(bs, (-1))
        tsz = tmp.shape[0]
        #print('labpath = %s , tsz = %d' % (labpath, tsz))
        if tsz > max_boxes*5:
            label = tmp[0 : max_boxes*5]
        elif tsz > 0:
            label[0:tsz] = tmp
    return label

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    
    label = fill_truth_detection(labpath, flip, dx, dy, 1./sx, 1./sy)
    return img,label
