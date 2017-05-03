import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j) > nms_thresh:
                    box_j[4] = 0
    return out_boxes

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_region_boxes(output, conf_thresh, num_classes, anchors):
    num_anchors = len(anchors)/2
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    if batch == 1:
        t0 =time.time()

        boxes = []
        output = output.view(num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, num_anchors*h*w)
        xy = torch.sigmoid(output[0:2])
        wh = torch.exp(output[2:4])
        det_confs = torch.sigmoid(output[4])
        cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data
        cls_confs, cls_ids = torch.max(cls_confs, 1)
        cls_confs = cls_confs.view(-1)
        cls_ids = cls_ids.view(-1)
        t1 =time.time()

        det_confs = convert2cpu(det_confs)
        xy = convert2cpu(xy)
        wh = convert2cpu(wh)
        cls_confs = convert2cpu(cls_confs)
        cls_ids = convert2cpu_long(cls_ids)
        t2 = time.time()
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = i*h*w + cy * w + cx
                    det_conf =  det_confs[ind]
    
                    if det_conf > conf_thresh:
                        bcx = xy[0][ind] + cx
                        bcy = xy[1][ind] + cy
                        bw = anchors[2*i] * wh[0][ind]
                        bh = anchors[2*i+1] * wh[1][ind]
                        cls_conf = cls_confs[ind]
                        cls_id = cls_ids[ind]
                        x1 = bcx - bw/2
                        y1 = bcy - bh/2
                        x2 = bcx + bw/2
                        y2 = bcy + bh/2
                        x1 = max(x1, 0.0)
                        y1 = max(y1, 0.0)
                        x2 = min(x2, w)
                        y2 = min(y2, h)
                        box = [x1/w, y1/h, x2/w, y2/h, det_conf, cls_conf, cls_id]
                        boxes.append(box)
        t3 = time.time()
        if False:
            print('---------------------------------')
            print('matrix computation : %f' % (t1-t0))
            print('        gpu to cpu : %f' % (t2-t1))
            print('      boxes filter : %f' % (t3-t2))
            print('---------------------------------')
        return boxes
    else:
        t0 = time.time()
        all_boxes = []
        output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)

        grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
        grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).cuda()
        xs = torch.sigmoid(output[0]) + grid_x
        ys = torch.sigmoid(output[1]) + grid_y

        anchor_x = torch.Tensor(anchors).view(num_anchors,2).index_select(1, torch.LongTensor([0]))
        anchor_y = torch.Tensor(anchors).view(num_anchors,2).index_select(1, torch.LongTensor([1]))
        anchor_x = anchor_x.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
        anchor_y = anchor_y.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).cuda()
        ws = torch.exp(output[2]) * anchor_x
        hs = torch.exp(output[3]) * anchor_y

        det_confs = torch.sigmoid(output[4])

        cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data
        cls_confs, cls_ids = torch.max(cls_confs, 1)
        cls_confs = cls_confs.view(-1)
        cls_ids = cls_ids.view(-1)
        t1 = time.time()
        
        sz_hw = h*w
        sz_hwa = sz_hw*num_anchors
        det_confs_cpu = torch.FloatTensor(det_confs.size()).copy_(det_confs)
        cls_confs_cpu = torch.FloatTensor(cls_confs.size()).copy_(cls_confs)
        cls_ids_cpu = torch.LongTensor(cls_ids.size()).copy_(cls_ids)
        xs_cpu = torch.FloatTensor(xs.size()).copy_(xs)
        ys_cpu = torch.FloatTensor(ys.size()).copy_(ys)
        ws_cpu = torch.FloatTensor(ws.size()).copy_(ws)
        hs_cpu = torch.FloatTensor(hs.size()).copy_(hs)
        t2 = time.time()
        for b in range(batch):
            boxes = []
            for cy in range(h):
                for cx in range(w):
                    for i in range(num_anchors):
                        ind = b*sz_hwa + i*sz_hw + cy*w + cx
                        det_conf =  det_confs_cpu[ind]
        
                        if det_conf > conf_thresh:
                            bcx = xs_cpu[ind]
                            bcy = ys_cpu[ind]
                            bw = ws_cpu[ind]
                            bh = hs_cpu[ind]
                            cls_conf = cls_confs_cpu[ind]
                            cls_id = cls_ids_cpu[ind]
                            x1 = bcx - bw/2
                            y1 = bcy - bh/2
                            x2 = bcx + bw/2
                            y2 = bcy + bh/2
                            x1 = max(x1, 0.0)
                            y1 = max(y1, 0.0)
                            x2 = min(x2, w)
                            y2 = min(y2, h)
                            box = [x1/w, y1/h, x2/w, y2/h, det_conf, cls_conf, cls_id]
                            boxes.append(box)
            all_boxes.append(boxes)
        t3 = time.time()
        if False:
            print('---------------------------------')
            print('matrix computation : %f' % (t1-t0))
            print('        gpu to cpu : %f' % (t2-t1))
            print('      boxes filter : %f' % (t3-t2))
            print('---------------------------------')
        return all_boxes

def plot_boxes(img, boxes, savename, class_names=None):
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]]);
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = box[0] * width
        y1 = box[1] * height
        x2 = box[2] * width
        y2 = box[3] * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline = rgb)
    img.save(savename)

def read_truths(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size/5, 5)
        return truths
    else:
        return np.array([])

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0,1).transpose(0,2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t2 = time.time()

    output = model(img)
    output = output.data
    t3 = time.time()

    boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors)
    t4 = time.time()

    boxes = nms(boxes, nms_thresh)
    t5 = time.time()

    if False:
        print('-----------------------------------')
        print(' image to tensor : %f' % (t1 - t0))
        print('  tensor to cuda : %f' % (t2 - t1))
        print('         predict : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print('             nms : %f' % (t5 - t4))
        print('           total : %f' % (t5 - t0))
        print('-----------------------------------')
    return boxes

