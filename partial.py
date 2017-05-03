from darknet import Darknet
from darknet2 import Darknet2

def partial(cfgfile, weightfile, outfile, cutoff):
    m = Darknet2(cfgfile)
    m.float()
    m.load_weights(weightfile)
    m.save_weights(outfile, cutoff)
    print('save %s' % (outfile))

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        outfile = sys.argv[3]
        cutoff = int(sys.argv[4])
        partial(cfgfile, weightfile, outfile, cutoff)
    else:
        partial('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'tiny-yolo-voc.conv.15', 15)
