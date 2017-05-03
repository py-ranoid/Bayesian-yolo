from PIL import Image, ImageDraw
from tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
from darknet2 import Darknet2

def demo1(tiny_yolo_weight, img_path):
    m = TinyYoloNet() 
    m.eval()
    m.load_darknet_weights(tiny_yolo_weight)
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(img_path).convert('RGB')
    sized = img.resize((416,416))
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

    class_names = load_class_names('data/voc.names')
    plot_boxes(img, boxes, 'predict1.jpg', class_names)  

def demo2(cfgfile, weightfile, img_path):
    m = Darknet(cfgfile) 
    m.load_weights(weightfile)
    m.eval()
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(img_path).convert('RGB')
    sized = img.resize((416,416))
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

    class_names = load_class_names('data/voc.names')
    plot_boxes(img, boxes, 'predict2.jpg', class_names)

def demo3(cfgfile, weightfile, img_path):
    m = Darknet2(cfgfile) 
    m.load_weights(weightfile)
    m.eval()
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(img_path).convert('RGB')
    sized = img.resize((416,416))
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

    class_names = load_class_names('data/voc.names')
    plot_boxes(img, boxes, 'predict3.jpg', class_names)

def demo4(cfgfile, weightfile, videofile):
    m = Darknet2(cfgfile) 
    m.load_weights(weightfile)
    m.eval()
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(img_path).convert('RGB')
    sized = img.resize((416,416))
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

    class_names = load_class_names('data/voc.names')
    plot_boxes(img, boxes, 'predict3.jpg', class_names)

############################################
if __name__ == '__main__':
    demo1('tiny-yolo-voc.weights', 'data/person.jpg')
    demo2('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg')
    demo3('cfg/yolo-voc.cfg', 'yolo-voc.weights', 'data/person.jpg')
