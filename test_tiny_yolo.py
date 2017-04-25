from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn1 = nn.Sequential(
            # conv1
            nn.Conv2d( 3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # conv2
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # conv3
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # conv4
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # conv5
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # conv6
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.cnn2 = nn.Sequential(
            # conv7
            nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # conv8
            nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            # output
            nn.Conv2d(1024, 125, 1, 1, 0),
        )

    def forward(self, x):
        x = F.max_pool2d(F.pad(self.cnn1(x), (0,1,0,1), mode='replicate'), 2, stride=1)
        x = self.cnn2(x)
        return x
        #return F.log_softmax(x)

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

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def get_region_boxes(output, thresh):
    num_classes = 20
    num_anchors = 5
    anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
    assert(output.size(0) == 1)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)
    boxes = []
    for cy in range(h):
        for cx in range(w):
            for i in range(num_anchors):
                start = (5+num_classes)*i
                bcx = sigmoid(output[0][start][cy][cx]) + cx
                bcy = sigmoid(output[0][start+1][cy][cx]) + cy
                bw = anchors[2*i] * math.exp(output[0][start+2][cy][cx])
                bh = anchors[2*i+1] * math.exp(output[0][start+3][cy][cx])
                det_conf = sigmoid(output[0][start+4][cy][cx]) 
                x1 = bcx - bw/2
                y1 = bcy - bh/2
                x2 = bcx + bw/2
                y2 = bcy + bh/2
                if det_conf > thresh:
                    box = [x1/w, y1/h, x2/w, y2/h, det_conf]
                    boxes.append(box)
    return boxes
                

def do_detect(model, img):
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
    boxes = get_region_boxes(output, 0.6)
    #boxes = nms(boxes, 0.4)
    return boxes
    
def plot_boxes(img, boxes):
    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = box[0] * width
        y1 = box[1] * height
        x2 = box[2] * width
        y2 = box[3] * height
        draw.rectangle([x1, y1, x2, y2])
    img.save('predict.png')

m = Net() 
m.float()
m.eval()

buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
start = 4

start = load_conv_bn(buf, start, m.cnn1[0], m.cnn1[1])
start = load_conv_bn(buf, start, m.cnn1[4], m.cnn1[5])
start = load_conv_bn(buf, start, m.cnn1[8], m.cnn1[9])
start = load_conv_bn(buf, start, m.cnn1[12], m.cnn1[13])
start = load_conv_bn(buf, start, m.cnn1[16], m.cnn1[17])
start = load_conv_bn(buf, start, m.cnn1[20], m.cnn1[21])

start = load_conv_bn(buf, start, m.cnn2[0], m.cnn2[1])
start = load_conv_bn(buf, start, m.cnn2[3], m.cnn2[4])
start = load_conv(buf, start, m.cnn2[6])

img = Image.open('person.jpg').convert('RGB')
boxes = do_detect(m, img)
plot_boxes(img, boxes)
