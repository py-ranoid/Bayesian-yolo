### pytorch-yolo2
Convert https://pjreddie.com/darknet/yolo/ into pytorch2. This repository is trying to achieve the following goals.
- detect, partial, valid functions
- load darknet cfg
- load darknet saved weights
- save as darknet weights
- fast evaluation
- fast training
- implement RegionLoss, MaxPoolStrid1, Reorg, GolbalAvgPool2d without cffi bindings

---
#### Detection Using A Pre-Trained Model
```
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg yolo.weights data/dog.jpg
```
You will see some output like this:
```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32
    1 max          2 x 2 / 2   416 x 416 x  32   ->   208 x 208 x  32
    ......
   30 conv    425  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 425
   31 detection
Loading weights from yolo.weights... Done!
data/dog.jpg: Predicted in 0.014079 seconds.
truck: 0.926668
bicycle: 0.997449
dog: 0.993459
```
---
#### Real-Time Detection on a Webcam

---

#### Training YOLO on VOC
##### Get The Pascal VOC Data
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
##### Generate Labels for VOC
```
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
```
##### Modify Cfg for Pascal Data
Change the cfg/voc.data config file
```
train  = train.txt
valid  = 2007_test.txt
names = data/voc.names
backup = backup
```
##### Download Pretrained Convolutional Weights
Download weights from the convolutional layers
```
wget http://pjreddie.com/media/files/darknet19_448.conv.23
```
or run the following command:
```
python partial.py cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23
```
##### Train The Model
```
python train.py cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
##### Evaluate The Model
```
python valid.py cfg/voc.data cfg/yolo-voc.cfg ../yolo-voc.weights
```
---
#### Todo
- [x] support avgpool softmax sse layer in darknet2.py
- [x] verify python partial.py cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23
- [ ] compare with darknet to make same forward softmax results by modeify batch_norm layer
---
#### Problems
##### 1. Running variance difference between darknet and pytorch
Change the code in normalize_cpu to make the same result
```
normalize_cpu:
x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .00001f));
``` 
##### 2. Predition is very time consuming
The first forward is usually slow, you can forward many times to test the speed.
