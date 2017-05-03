import sys
import time
from PIL import Image, ImageDraw
from tiny_yolo import TinyYoloNet
from utils import do_detect, plot_boxes, load_class_names
from darknet import Darknet
from darknet2 import Darknet2

def detect(cfgfile, weightfile, imgfile, version=2):
    if version == 1:
        m = Darknet(cfgfile) 
    elif version == 2:
        m = Darknet2(cfgfile) 

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        detect(cfgfile, weightfile, imgfile, version=2)
    else:
        detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
