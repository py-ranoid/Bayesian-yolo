import math
import torch
from PIL import Image, ImageDraw

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def bbox_iou(bbox1, bbox2):
    mx = min(bbox1[0], bbox2[0])
    Mx = max(bbox1[2], bbox2[2])
    my = min(bbox1[1], bbox2[1])
    My = max(bbox1[3], bbox2[3])
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]
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

def nms(boxes, thresh):
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
                if bbox_iou(box_i, box_j) > thresh:
                    box_j[4] = 0
    return out_boxes

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


