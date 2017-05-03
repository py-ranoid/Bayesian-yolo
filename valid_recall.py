from PIL import Image, ImageDraw
from utils import *
from tiny_yolo_face14 import TinyYoloFace14Net
from darknet import Darknet

def eval_list(cfgfile, weightfile, img_list, eval_wid, eval_hei):
    #m = TinyYoloFace14Net()
    #m.eval()
    #m.load_darknet_weights(tiny_yolo_weight)

    m = Darknet(cfgfile)
    m.eval()
    m.load_weights(weightfile)

    use_cuda = 1
    if use_cuda:
        m.cuda()

    conf_thresh = 0.25
    nms_thresh = 0.4
    iou_thresh = 0.5

    with open(img_list) as fp:
        lines = fp.readlines()

    total = 0.0
    proposals = 0.0
    correct = 0.0
    lineId = 0
    for line in lines:
        lineId = lineId + 1
        img_path = line.rstrip()
        lab_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        truths = read_truths(lab_path)
        #print(truths)

        img = Image.open(img_path).convert('RGB').resize((eval_wid, eval_hei))
        boxes = do_detect(m, img, conf_thresh, nms_thresh, use_cuda)
        savename = "tmp/%06d.jpg" % (lineId)
        print("save %s" % savename)
        plot_boxes(img, boxes, savename)
        
        total = total + truths.shape[0]

        for i in range(len(boxes)):
            if boxes[i][4] > conf_thresh:
                proposals = proposals+1

        for i in range(truths.shape[0]):
            x1 = truths[i][1] - truths[i][3]/2.0
            y1 = truths[i][2] - truths[i][4]/2.0
            x2 = truths[i][1] + truths[i][3]/2.0
            y2 = truths[i][2] + truths[i][4]/2.0
            box_gt = [x1, y1, x2, y2, 1.0]
            best_iou = 0
            for j in range(len(boxes)):
                iou = bbox_iou(box_gt, boxes[j])
                best_iou = max(iou, best_iou)
            if best_iou > iou_thresh:
                correct = correct+1

        precision = 1.0*correct/proposals
        recall = 1.0*correct/total
        fscore = 2.0*precision*recall/(precision+recall)
        print("%d precision: %f, recal: %f, fscore: %f\n" % (lineId, precision, recall, fscore))

if __name__ == '__main__':
    #eval_list('face4.1nb_inc2_96.16.weights', 'test.txt', 160, 160)
    eval_list('face4.1nb_inc2_96.16_nobn.cfg', 'face4.1nb_inc2_96.16_nobn.weights', 'test.txt', 160, 160)
    #eval_list('face4.1nb_inc2_96.16.cfg', 'face4.1nb_inc2_96.16.weights', 'test.txt', 160, 160)
