import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import bbox_iou

def build_targets(target, anchors, nW, nH):
    nB = target.size(0)
    nA = len(anchors)/2
    mask = torch.zeros(nB, nA, nW, nH)
    tx   = torch.zeros(nB, nA, nW, nH) 
    ty   = torch.zeros(nB, nA, nW, nH) 
    tw   = torch.zeros(nB, nA, nW, nH) 
    th   = torch.zeros(nB, nA, nW, nH) 
    tconf = torch.zeros(nB, nA, nW, nH)

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
                anchor_box = [0, 0, anchors[2*n], anchors[2*n+1]]
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
            mask[b][best_n][j][i] = 1
            tx[b][best_n][j][i] = target[b][t*5+1] * nW - i
            ty[b][best_n][j][i] = target[b][t*5+2] * nH - j
            tw[b][best_n][j][i] = math.log(w/anchors[2*best_n])
            th[b][best_n][j][i] = math.log(h/anchors[2*best_n+1])
            tconf[b][best_n][j][i] = best_iou

    return nGT, mask, tx, ty, tw, th, tconf

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[]):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(self.anchors)/2
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        nGT, mask, tx, ty, tw, th, tconf = build_targets(target.data, self.anchors, nW, nH)
        scale_mask = self.noobject_scale * (1-mask) + self.object_scale * mask

        tx   = Variable(tx.cuda())
        ty   = Variable(ty.cuda())
        tw   = Variable(tw.cuda())
        th   = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        mask = Variable(mask.cuda())
        scale_mask = Variable(scale_mask.cuda())

        output = output.view(nB, nA, (5+nC), nH, nW)

        x = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*mask, tx*mask) #/nGT
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*mask, ty*mask) #/nGT
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*mask, tw*mask) #/nGT
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*mask, th*mask) #/nGT
        loss_conf = nn.MSELoss(size_average=False)(conf*scale_mask, tconf*scale_mask) #/nGT
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf
        return loss
