import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def parse_prototxt(protofile):
    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_param_block(fp):
        block = dict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                block[key] = value
            elif ltype == 1: # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_param_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
        return block

    def parse_layer_block(fp):
        block = dict()
        block['top'] = []
        block['bottom'] = []
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if key == 'top' or key == 'bottom':
                    block[key].append(value)
                else:
                    block[key] = value
            elif ltype == 1: # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_param_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
        return block

    fp = open(protofile, 'r')
    props = dict()
    layers = []
    line = fp.readline()
    while line != '':
        ltype = line_type(line)
        if ltype == 0: # key: value
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            props[key] = value
        elif ltype == 1: # blockname {
            key = line.split('{')[0].strip()
            assert(key == 'layer')
            layer = parse_layer_block(fp)
            layers.append(layer)
        line = fp.readline()
    net_info = dict()
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info


class CaffeNet(nn.Module):
    def __init__(self, protofile):
        super(CaffeNet, self).__init__()
        self.seen = 0
        self.num_classes = 20
        self.anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
        self.num_anchors = len(self.anchors)/2
        self.width = 160
        self.height = 160

        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)

        self.net_info = parse_prototxt(protofile)
        self.models = create_network(self.net_info)


    def print_network(self):
        print(self.net_info)

    def create_network(self, net_info):
        models = nn.Sequential()

        prev_filters = 1
        for layer in net_info['layers']:
            name = layer['name']
            ltype = layer['type']
            if ltype == 'Data':
                continue
            elif ltype == 'Convolution':
                filters = int(layer['convolution_param']['num_output'])
                kernel_size = int(layer['convolution_param']['kernel_size'])
                stride = layer['convolution_param']['stride']
                stride = int(stride) if stride or 1
                pad = layer['convolution_param']['pad']
                pad = int(pad) if pad or 0
                group = layer['convolution_param']['group']
                group = int(group) if group or 1
                models.add_module(name, nn.Conv2d(prev_filters, filters, kernel_size, stride,pad,groups=group))
                prev_filters = filters
            elif ltype == 'ReLU':
                bottom = layer['bottom']
                top = layer['top']
                inplace = (bottom == top)
                models.add_module(name, nn.ReLU(inplace=inplace))
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = int(layer['pooling_param']['stride'])
                models.add_module(name, nn.Maxpool2d(kernel_size, stride))
        return models
