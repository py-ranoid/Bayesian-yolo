from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from collections import OrderedDict
from utils import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn1 = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d( 3, 16, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),

            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),

            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),

            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),

            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('leaky5', nn.LeakyReLU(0.1, inplace=True)),
            ('pool5', nn.MaxPool2d(2, 2)),

            # conv6
            ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),
        ]))

        self.cnn2 = nn.Sequential(OrderedDict([
            # conv7
            ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('leaky7', nn.LeakyReLU(0.1, inplace=True)),

            # conv8
            ('conv8', nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(1024)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),

            # output
            ('output', nn.Conv2d(1024, 125, 1, 1, 0)),
        ]))

    def forward(self, x):
        x = F.max_pool2d(F.pad(self.cnn1(x), (0,1,0,1), mode='replicate'), 2, stride=1)
        x = self.cnn2(x)
        return x
        #return F.log_softmax(x)

    def load_darknet_weights(self, path):
        def load_conv(buf, start, conv_model):
            num_w = conv_model.weight.numel()
            num_b = conv_model.bias.numel()
            conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
            conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
            return start
        
        def load_conv_bn(buf, start, conv_model, bn_model):
            num_w = conv_model.weight.numel()
            num_b = bn_model.bias.numel()
            bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
            bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
            bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
            running_var = torch.from_numpy(buf[start:start+num_b]); start = start + num_b
            bn_model.running_var.copy_((running_var.sqrt() + .00001).pow(2) - 0.00001)
            conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w 
            return start
        
        def load_fc(buf, start, fc_model):
            num_w = fc_model.weight.numel()
            num_b = fc_model.bias.numel()
            fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
            fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w 
            return start

        #buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
        buf = np.fromfile(path, dtype = np.float32)
        start = 4
        
        start = load_conv_bn(buf, start, self.cnn1[0], self.cnn1[1])
        start = load_conv_bn(buf, start, self.cnn1[4], self.cnn1[5])
        start = load_conv_bn(buf, start, self.cnn1[8], self.cnn1[9])
        start = load_conv_bn(buf, start, self.cnn1[12], self.cnn1[13])
        start = load_conv_bn(buf, start, self.cnn1[16], self.cnn1[17])
        start = load_conv_bn(buf, start, self.cnn1[20], self.cnn1[21])
        
        start = load_conv_bn(buf, start, self.cnn2[0], self.cnn2[1])
        start = load_conv_bn(buf, start, self.cnn2[3], self.cnn2[4])
        start = load_conv(buf, start, self.cnn2[6])


def do_detect(model, img, thresh, nms_thresh):
    model.eval()
    img = img.resize((416, 416))
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    img = Variable(img)

    output = model(img)
    output = output.data
    boxes = get_region_boxes(output, thresh)
    boxes = nms(boxes, nms_thresh)
    return boxes
    
############################################
m = Net() 
m.float()
m.eval()
m.load_darknet_weights('tiny-yolo-voc.weights')

img = Image.open('person.jpg').convert('RGB')
boxes = do_detect(m, img, 0.5, 0.4)
plot_boxes(img, boxes)
