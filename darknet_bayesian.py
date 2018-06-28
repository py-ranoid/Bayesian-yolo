import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from region_loss import RegionLoss
from cfg import *
import BayesianLayers2 as BayesianLayers
# import BayesianLayers
# from compression import compute_compression_rate, compute_reduced_weights

#from layers.batchnorm.bn import BN2d

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H/hs, W/ws)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky

        self.kl_list = [self.models[0].conv1, self.models[2].conv2,self.models[4].conv3,self.models[6].conv4,self.models[8].conv5,self.models[10].conv6,self.models[12].conv7,self.models[13].conv8,self.models[14].conv9]
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def get_masks(self, thresholds):
        weight_masks = []
        bias_masks = []
        conv_mask = None
        lin_mask = None
        for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
            # compute dropout mask
            if layer.get_type() == 'conv':
                if conv_mask is None:
                    mask = [True] * layer.in_channels
                else:
                    mask = np.copy(conv_mask)

                # print ("CONV:", np.array(mask).shape)
                log_alpha = layers[i].get_log_dropout_rates(
                ).cpu().data.numpy()
                conv_mask = log_alpha < thresholds[i]
                # print ("CONV-MASK:", conv_mask.shape)
                # print (layer.bias_mu.shape)

                # print(np.sum(mask), np.sum(conv_mask))

                weight_mask = np.expand_dims(
                    mask, axis=0) * np.expand_dims(conv_mask, axis=1)
                weight_mask = weight_mask[:, :, None, None]
                bias_mask = conv_mask
            else:
                if lin_mask is None:
                    mask = conv_mask.repeat(
                        layer.in_features / conv_mask.shape[0])
                else:
                    mask = np.copy(lin_mask)
                # print ("LIN:", mask.shape)
                try:
                    log_alpha = layers[i +
                                       1].get_log_dropout_rates().cpu().data.numpy()
                    lin_mask = log_alpha < thresholds[i + 1]
                except:
                    # must be the last mask
                    lin_mask = np.ones(10)
                # print ("LIN-MASK:", lin_mask.shape)
                # print (layer.bias_mu.shape)
                # print(np.sum(mask), np.sum(lin_mask))

                weight_mask = np.expand_dims(
                    mask, axis=0) * np.expand_dims(lin_mask, axis=1)
                bias_mask = lin_mask

            weight_masks.append(weight_mask.astype(np.float))
            bias_masks.append(bias_mask.astype(np.float))
        return weight_masks, bias_masks

        # def get_masks(self, thresholds):
        #     weight_masks = []
        #     bias_masks = []
        #     conv_mask = None
        #     lin_mask = None
        #     for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
        #         # compute dropout mask
        #         if layer.get_type() == 'conv':
        #             if conv_mask is None:
        #                 mask = [True] * layer.in_channels
        #             else:
        #                 mask = np.copy(conv_mask)
        #
        #             # print ("CONV:", np.array(mask).shape)
        #             log_alpha = layers[i].get_log_dropout_rates(
        #             ).cpu().data.numpy()
        #             conv_mask = log_alpha < thresholds[i]
        #             # print ("CONV-MASK:", conv_mask.shape)
        #             # print (layer.bias_mu.shape)
        #
        #             # print(np.sum(mask), np.sum(conv_mask))
        #
        #             weight_mask = np.expand_dims(
        #                 mask, axis=0) * np.expand_dims(conv_mask, axis=1)
        #             weight_mask = weight_mask[:, :, None, None]
        #             bias_mask = conv_mask
        #         else:
        #             if lin_mask is None:
        #                 mask = conv_mask.repeat(
        #                     layer.in_features / conv_mask.shape[0])
        #             else:
        #                 mask = np.copy(lin_mask)
        #             # print ("LIN:", mask.shape)
        #             try:
        #                 log_alpha = layers[i +
        #                                    1].get_log_dropout_rates().cpu().data.numpy()
        #                 lin_mask = log_alpha < thresholds[i + 1]
        #             except:
        #                 # must be the last mask
        #                 lin_mask = np.ones(10)
        #             # print ("LIN-MASK:", lin_mask.shape)
        #             # print (layer.bias_mu.shape)
        #             # print(np.sum(mask), np.sum(lin_mask))
        #
        #             weight_mask = np.expand_dims(
        #                 mask, axis=0) * np.expand_dims(lin_mask, axis=1)
        #             bias_mask = lin_mask
        #
        #         weight_masks.append(weight_mask.astype(np.float))
        #         bias_masks.append(bias_mask.astype(np.float))
        #     return weight_masks, bias_masks

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD


    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or block['type'] == 'maxpool' or block['type'] == 'reorg' or block['type'] == 'avgpool' or block['type'] == 'softmax' or block['type'] == 'connected':
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
        # print (models)

        prev_filters = 3
        out_filters =[]
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)/2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    # model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))

                    model.add_module('conv{0}'.format(conv_id), BayesianLayers.Conv2dGroupNJ(prev_filters, filters, kernel_size,stride,pad,bias=False,cuda=True))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    # model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                    model.add_module('conv{0}'.format(conv_id), BayesianLayers.Conv2dGroupNJ(prev_filters, filters, kernel_size,stride,pad,cuda=True))


                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    # model = nn.Linear(prev_filters, filters)
                    model =  BayesianLayers.LinearGroupNJ(prev_filters,filters,cuda = True)

                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)/loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            else:
                print('unknown type %s' % (block['type']))


        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                # print batch_normalize
                if batch_normalize:
                    conv_model = model[0]
                    bn_model = model[1]
                    # start = load_conv_bn(buf, start, model[0], model[1])
                    num_w = conv_model.weight_mu.numel()
                    num_b = bn_model.bias.numel()
                    # bn_model.bias.data.copy_(torch.reshape(torch.from_numpy(buf[start:start+num_w]),(bn_model.bias.shape[0],bn_model.bias.shape[1], bn_model.bias.shape[2],bn_model.bias.shape[3])))
                    # start = start + num_b
                    # bn_model.weight.data.copy_(torch.reshape(torch.from_numpy(buf[start:start+num_w]),(bn_model.weight.shape[0],bn_model.weight.shape[1], bn_model.weight.shape[2],bn_model.weight.shape[3])))
                    # start = start + num_b
                    # bn_model.running_mean.copy_(torch.reshape(torch.from_numpy(buf[start:start+num_w]),(bn_model.running_mean.shape[0],bn_model.running_mean.shape[1], bn_model.running_mean.shape[2],bn_model.running_mean.shape[3])));
                    # start = start + num_b
                    # bn_model.running_var.copy_(torch.reshape(torch.from_numpy(buf[start:start+num_w]),(bn_model.running_var.shape[0],bn_model.running_var.shape[1], bn_model.running_var.shape[2],bn_model.running_var.shape[3])));
                    # start = start + num_b
                    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
                    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
                    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
                    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b

                    conv_model.weight_mu.data.copy_(torch.reshape(torch.from_numpy(buf[start:start+num_w]),(conv_model.weight_mu.shape[0],conv_model.weight_mu.shape[1], conv_model.weight_mu.shape[2],conv_model.weight_mu.shape[3])));
                    start = start + num_w

                else:
                    conv_model = model[0]
                    num_w = conv_model.weight_mu.numel()
                    num_b = conv_model.bias_mu.numel()
                    conv_model.bias_mu.data.copy_(torch.from_numpy(buf[start:start+num_b]));
                    start = start + num_b
                    conv_model.weight_mu.data.copy_(torch.reshape(torch.from_numpy(buf[start:start+num_w]),(conv_model.weight_mu.shape[0],conv_model.weight_mu.shape[1], conv_model.weight_mu.shape[2],conv_model.weight_mu.shape[3])));
                    start = start + num_w
                #     start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()
