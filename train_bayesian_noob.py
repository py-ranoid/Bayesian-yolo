from __future__ import print_function
import sys
import time
import os

BASE_PATH = '/gandiva-store/user1/pytorch-yolo2/' if os.environ.get('GANDIVA_USER') else './'
if os.environ.get('GANDIVA_USER',None):
    log_path = BASE_PATH + 'logs/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log_file = open(log_path+'log_'+str(int(time.time())),"w")
    sys.stdout = log_file
    sys.stderr = log_file

if len(sys.argv) != 4:
    print('Usage:')
    print('python train.py datacfg cfgfile weightfile')
    exit()

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

import dataset
import numpy as np
from utils import *
from cfg import parse_cfg,save_cfg
# from darknet import Darknet
from matplotlib import pyplot as plt
from darknet_bayesian import Darknet
from compression2 import compute_compression_rate, compute_reduced_weights

# from compression import compute_compression_rate, compute_reduced_weights

# Training settings
datacfg       = sys.argv[1]
cfgfile       = sys.argv[2]
weightfile    = sys.argv[3]


data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]

trainlist     = data_options['train']
testlist      = data_options['valid']
backupdir     = data_options['backup']
nsamples      = file_lines(trainlist)
# print (nsamples)
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
max_epochs    = int(max_batches*batch_size/nsamples)+1
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
save_interval = 10  # epoches
dot_interval  = 70  # batches

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

if not os.path.exists(backupdir):
    os.mkdir(backupdir)

###############
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = Darknet(cfgfile)
region_loss = model.loss

model.load_weights(weightfile)
model.print_network()

region_loss.seen  = model.seen
processed_batches = model.seen/batch_size

init_width        = model.width
init_height       = model.height
init_epoch        = model.seen/nsamples

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       lambda x: 2 * (x - 0.5)
                   ]), train=False),
    batch_size=batch_size, shuffle=False, **kwargs)

# if use_cuda:
#     if ngpus > 1:
#         model = torch.nn.DataParallel(model).cuda()
#     else:
#         model = model.cuda()


model = model.cuda()
params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay*batch_size}]

optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate/batch_size,weight_decay=decay*batch_size)


print ([model.models[0].conv1, model.models[2].conv2,model.models[4].conv3,model.models[6].conv4,model.models[8].conv5,model.models[10].conv6,model.models[12].conv7,model.models[13].conv8,model.models[14].conv9])


def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr


# we optimize the variational lower bound scaled by the number of data
    # points (so we can keep our intuitions about hyper-params such as the learning rate)
# discrimination_loss = nn.functional.cross_entropy

def objective(output, target, kl_divergence):
        # discrimination_error = discrimination_loss(output, target)
        discrimination_error = region_loss(output, target)
        variational_bound = discrimination_error + kl_divergence / nsamples
        # if FLAGS.cuda:
        variational_bound = variational_bound.cuda()
        return variational_bound

def train(epoch):
    global processed_batches
    t0 = time.time()
    # if ngpus > 1:
    #     cur_model = model.module
    # else:

    cur_model = model
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                       shuffle=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           lambda x: 2 * (x - 0.5)
                       ]),
                       train=True,
                       seen=cur_model.seen,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)
    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        #if (batch_idx+1) % dot_interval == 0:
        #    sys.stdout.write('.')

        if use_cuda:
            data = data.cuda()
            #target= target.cuda()
        t3 = time.time()
        data, target = Variable(data), Variable(target)
        t4 = time.time()
        optimizer.zero_grad()
        t5 = time.time()
        output = model(data)
        t6 = time.time()
        region_loss.seen = region_loss.seen + data.data.size(0)

        # loss = region_loss(output, target)
        loss = objective(output, target, model.kl_divergence())


        t7 = time.time()
        loss.backward()
        t8 = time.time()
        optimizer.step()
        t9 = time.time()

        for layer in model.kl_list:
                layer.clip_variances()

        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
    print('')
    t1 = time.time()
    # logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    # if (epoch+1) % save_interval == 0:
    #     logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
    #     cur_model.seen = (epoch + 1) * len(train_loader.dataset)
    #     cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))


def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    # if ngpus > 1:
    #     cur_model = model.module
    # else:
    cur_model = model
    num_classes = cur_model.num_classes
    anchors     = cur_model.anchors
    num_anchors = cur_model.num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)

            total = total + num_gts

            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))

evaluate = False
if evaluate:
    logging('evaluating ...')
    test(0)
else:
    print ("MAX EPOCHS :",max_epochs)
    for epoch in range(init_epoch, max_epochs):
        break
    # for epoch in range(0, 5):
        train(epoch)
        test(epoch)

    vals_path = BASE_PATH+'vals'
    if not os.path.exists(vals_path):
        os.mkdir(vals_path)
    print ("MODEL SAVE TO PICKLE")

    pickle_fname = vals_path + "/ep_"+str(max_epochs)+"_darknet_bayes.pkl"
    torch.save(model.state_dict(), pickle_fname)
    print ("model")

    layers = [model.models[0].conv1, model.models[2].conv2,model.models[4].conv3,model.models[6].conv4,model.models[8].conv5,model.models[10].conv6,model.models[12].conv7,model.models[13].conv8,model.models[14].conv9]
    thresholds = [-3,-3,-3,-3,-3,-3,-3,-3,-3]
    log_alphas = [l.get_log_dropout_rates() for l in layers]

    thresh_path = BASE_PATH+'thresh_plots'
    if not os.path.exists(thresh_path):
        os.mkdir(thresh_path)
    for lr,lar in enumerate(log_alphas):
        plt.hist(lar.cpu().data.numpy(), bins=80)
        plt.xlabel('Threshold')
        plt.ylabel('# Groups')
        plt.savefig(thresh_path+'/lenet_%d_epochs_layer%d' %
                    (max_epochs, lr))
        # plt.show()
        plt.close()




    sig_bits,exp_bits = compute_compression_rate(layers, model.get_masks(thresholds))

    print("Test error after with reduced bit precision:")

    weights,biases = compute_reduced_weights(layers,model.get_masks(thresholds),
                                             sig_bits,exp_bits)
    all_files = []

    for i, (layer, weight, bias, block_i) in enumerate(zip(layers, weights, biases,model.conv_blocks_indices)):
        layer.post_weight_mu = torch.Tensor(weight).cuda()
        layer.post_bias_mu = (torch.Tensor(bias).cuda())
        model.blocks[block_i]['filters'] = int(layer.post_bias_mu.shape[0])

    paths,vals_path = model.save_weights_txt(max_epochs,vals_path,after=True)
    save_cfg(model.blocks,vals_path+'/output.cfg')
    header_gen(paths,vals_path +'/header_ep' + str(max_epochs)+'.h')

    for layer in layers: layer.deterministic = True

    # test(0)

if os.environ.get('GANDIVA_USER',None):
    log_file.close()
