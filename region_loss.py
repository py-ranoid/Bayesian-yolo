import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import bbox_iou

def build_targets(target, anchors, num_anchors, nH, nW):
    nB = target.size(0)
    nA = num_anchors
    anchor_step = len(anchors)/num_anchors
    mask  = torch.zeros(nB, nA, nH, nW)
    tx    = torch.zeros(nB, nA, nH, nW) 
    ty    = torch.zeros(nB, nA, nH, nW) 
    tw    = torch.zeros(nB, nA, nH, nW) 
    th    = torch.zeros(nB, nA, nH, nW) 
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls  = torch.zeros(nB, nA, nH, nW) 

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

            mask[b][best_n][j][i] = 1
            tx[b][best_n][j][i] = target[b][t*5+1] * nW - i
            ty[b][best_n][j][i] = target[b][t*5+2] * nH - j
            tw[b][best_n][j][i] = math.log(w/anchors[anchor_step*best_n])
            th[b][best_n][j][i] = math.log(h/anchors[anchor_step*best_n+1])
            tconf[b][best_n][j][i] = best_iou
            tcls[b][best_n][j][i] = target[b][t*5]

    return nGT, mask, tx, ty, tw, th, tconf, tcls

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

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        nGT, mask, tx, ty, tw, th, tconf,tcls = build_targets(target.data, self.anchors, self.num_anchors, nH, nW)
        scale_mask = self.noobject_scale * (1-mask) + self.object_scale * mask
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

        output = output.view(nB, nA, (5+nC), nH, nW)

        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls  = output.index_select(2, Variable(torch.linspace(4,4+nC-1,nC).long().cuda()))
        cls  = cls.view(nB, nA, nC, nH, nW).view(nB*nA, nC, nH*nW).transpose(0,1).contiguous()
        cls  = cls.view(nC,nB*nA*nH*nW).transpose(0,1).contiguous().view(nB*nA*nH*nW, nC)

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*mask, tx*mask) /nGT
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*mask, ty*mask) /nGT
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*mask, tw*mask) /nGT
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*mask, th*mask) /nGT
        loss_conf = nn.MSELoss(size_average=False)(conf*scale_mask, tconf*scale_mask) /nGT
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls*cls_mask, tcls) /nGT
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        return loss
