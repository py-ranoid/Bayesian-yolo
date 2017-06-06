import os
import os.path
from PIL import Image
from darknet import Darknet
from utils import do_detect, plot_boxes, load_class_names

def plot_widerface(cfgfile, weightfile, valdir, savedir):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    use_cuda = 1
    if use_cuda:
        m.cuda()

    scale_size = 16
    class_names = load_class_names('data/names')
    for parent,dirnames,filenames in os.walk(valdir):
        if parent != valdir:
            for filename in filenames:
                imgfile = os.path.join(parent,filename)
                img = Image.open(imgfile).convert('RGB')
                sized_width = int(round(img.width*1.0/scale_size) * 16)
                sized_height = int(round(img.height*1.0/scale_size) * 16)
                sized = img.resize((sized_width, sized_height))
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                savename = os.path.join(savedir, filename)
                print('save to %s' % savename)
                plot_boxes(img, boxes, savename, class_names)


def eval_widerface(cfgfile, weightfile, valdir, savedir):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    use_cuda = 1
    if use_cuda:
        m.cuda()

    scale_size = 16
    class_names = load_class_names('data/names')
    for parent,dirnames,filenames in os.walk(valdir):
        if parent != valdir:
            for filename in filenames:
                imgfile = os.path.join(parent,filename)
                img = Image.open(imgfile).convert('RGB')
                sized_width = int(round(img.width*1.0/scale_size) * 16)
                sized_height = int(round(img.height*1.0/scale_size) * 16)
                sized = img.resize((sized_width, sized_height))
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                savename = os.path.join(savedir, filename)
                print('save to %s' % savename)
                plot_boxes(img, boxes, savename, class_names)

if __name__ == '__main__':
    plot_widerface('resnet50_test.cfg', 'resnet50_98000.weights', 'widerface/WIDER_val/images/', 'widerface/val_results/')
