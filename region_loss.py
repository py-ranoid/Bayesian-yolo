import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import sigmoid, bbox_iou, convert2cpu

def get_region_box(pred_boxes, b, n, j, i, nH, nW, anchors, anchor_step):
    x = i + sigmoid(pred_boxes[b][n][0][j][i])
    y = j + sigmoid(pred_boxes[b][n][1][j][i])
    w = math.exp(pred_boxes[b][n][2][j][i]) * anchors[anchor_step*n]
    h = math.exp(pred_boxes[b][n][3][j][i]) * anchors[anchor_step*n+1]
    return [x,y,w,h]

def build_targets(pred_boxes, target, anchors, num_anchors, nH, nW, noobject_scale, object_scale, sil_thresh):
    nB = target.size(0)
    nA = num_anchors
    anchor_step = len(anchors)/num_anchors
    scale_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
    mask       = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    th         = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    #sil_thresh = 0.6
    for b in range(nB):
        for j in range(nH):
            for i in range(nW):
                for n in range(nA):
                    pred_box = get_region_box(pred_boxes, b,n,j,i,nH,nW,anchors, anchor_step)
                    best_iou = 0
                    for t in range(50):
                        if target[b][t*5+1] == 0:
                            break
                        gx = target[b][t*5+1]*nW
                        gy = target[b][t*5+2]*nH
                        gw = target[b][t*5+3]*nW
                        gh = target[b][t*5+4]*nH
                        gt_box = [gx, gy, gw, gh]
                        iou = bbox_iou(pred_box, gt_box, x1y1x2y2=False)
                        best_iou = max(iou, best_iou)
                    if best_iou > sil_thresh:
                        scale_mask[b][n][j][i] = 0

    nGT = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            i = int(target[b][t*5+1] * nW)
            j = int(target[b][t*5+2] * nH)
            w = target[b][t*5+3]*nW
            h = target[b][t*5+4]*nH
            gt_box = [0, 0, w, h]
            for n in range(nA):
                anchor_box = [0, 0, anchors[anchor_step*n], anchors[anchor_step*n+1]]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n

            gt_box[0] = target[b][t*5+1] * nW
            gt_box[1] = target[b][t*5+2] * nH
            pred_box = get_region_box(pred_boxes, b, best_n, j, i, nH, nW, anchors, anchor_step)

            mask[b][best_n][j][i] = 1
            scale_mask[b][best_n][j][i] = object_scale
            tx[b][best_n][j][i] = target[b][t*5+1] * nW - i
            ty[b][best_n][j][i] = target[b][t*5+2] * nH - j
            tw[b][best_n][j][i] = math.log(w/anchors[anchor_step*best_n])
            th[b][best_n][j][i] = math.log(h/anchors[anchor_step*best_n+1])
            tconf[b][best_n][j][i] = best_iou # bbox_iou(gt_box, pred_box, x1y1x2y2=False) # best_iou
            tcls[b][best_n][j][i] = target[b][t*5]

    return nGT, mask, scale_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output   = output.view(nB, nA, (5+nC), nH, nW)
        pred_boxes = output.index_select(2, Variable(torch.cuda.LongTensor([0,1,2,3])))
        pred_boxes = convert2cpu(pred_boxes.data)

        nGT, mask, scale_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, num_anchors, \
                                                                   nH, nW, self.noobject_scale, self.object_scale, self.thresh)
        cls_mask = torch.stack([mask.view(-1)]*nC, 1)

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls  = Variable(tcls.view(-1).long().cuda())
        mask       = Variable(mask.cuda())
        scale_mask = Variable(scale_mask.cuda())
        cls_mask   = Variable(cls_mask.cuda())

        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*mask, tx*mask) /nGT
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*mask, ty*mask) /nGT
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*mask, tw*mask) /nGT
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*mask, th*mask) /nGT
        loss_conf = nn.MSELoss(size_average=False)(conf*scale_mask, tconf*scale_mask) /nGT
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls*cls_mask, tcls) /nGT
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        return loss
