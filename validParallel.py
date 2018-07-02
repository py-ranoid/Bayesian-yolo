from darknet_bayesian import Darknet
import dataset
import torch
from compression2 import compute_compression_rate,compute_reduced_weights
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
import os

def valid(datacfg, cfgfile, weightfile, outfile):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    name_list = options['names']
    prefix = 'results'
    names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    m = Darknet(cfgfile)

    # m.load_state_dict(torch.load("darknet_bayes_60.pkl"))
    m = torch.nn.DataParallel(m).cuda()
    m.load_state_dict(torch.load(weightfile))
    m.module.print_network()
    # m.load_weights(weightfile)

    # m.cuda()
    m.module.eval()

#################################################
    """
    layers = [m.module.models[0].conv1,
              m.module.models[2].conv2,
              m.module.models[4].conv3,
              m.module.models[6].conv4,
              m.module.models[8].conv5,
              m.module.models[10].conv6,
              m.module.models[12].conv7,
              m.module.models[13].conv8,
              m.module.models[14].conv9]
    thresholds = [-3,-3,-3,-3,-3,-3,-3,-3,-3]
    sig,exp = compute_compression_rate(layers, m.module.get_masks(thresholds))


    print("Test error after with reduced bit precision:")

    weights,biases = compute_reduced_weights(layers, m.module.get_masks(thresholds),sig,exp)
    for layer, weight,bias in zip(layers, weights,biases):
        if True:
            print (layer.post_weight_mu.shape)
            print (weight.shape)
            layer.post_weight_mu.data = torch.Tensor(weight).cuda()
            layer.post_bias_mu = torch.Tensor(bias).cuda()
        else:
            layer.post_weight_mu.data = torch.Tensor(weight)
    for layer in layers: layer.deterministic = True
    """
    
##########################################

    valid_dataset = dataset.listDataset(valid_images, shape=(m.module.width, m.module.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs)

    fps = [0]*m.module.num_classes
    if not os.path.exists('results'):
        os.mkdir('results')
    for i in range(m.module.num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps[i] = open(buf, 'w')

    lineId = -1

    conf_thresh = 0.005
    nms_thresh = 0.45
    for batch_idx, (data, target) in enumerate(valid_loader):
        data = data.cuda()
        data = Variable(data, volatile = True)
        output = m(data).data
        batch_boxes = get_region_boxes(output, conf_thresh, m.module.num_classes, m.module.anchors, m.module.num_anchors, 0, 1)
        for i in range(output.size(0)):
            lineId = lineId + 1
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            width, height = get_image_size(valid_files[lineId])
            print(valid_files[lineId])
            boxes = batch_boxes[i]
            boxes = nms(boxes, nms_thresh)
            for box in boxes:
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height

                det_conf = box[4]
                for j in range(int((len(box)-5)/2)):
                    cls_conf = box[5+2*j]
                    cls_id = box[6+2*j]
                    prob =det_conf * cls_conf
                    fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))

    for i in range(m.module.num_classes):
        fps[i].close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
