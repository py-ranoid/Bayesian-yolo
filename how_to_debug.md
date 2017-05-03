#### Pytorch-yolo2
1. In darknet2.py, uncomment if condition to stop forward on certain layer
```
    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 27:
            #    break
```
2. In utils.py:do_detect, add
```
print(output.storage()[0:100])
```
to get the layer output

3. Prepare a test image, resized to 416x416, save to test.png. Stop resize
in detect.py:detect
```
sized = img.resize((m.width, m.height)) ->
sized = img
```

#### Darknet
1. In src/detector.c:test_dector, stop resize
```
image sized = letterbox_image(im, net.w, net.h);  ->
image sized = im
```
And add output print
```
layer ll = net.layers[27];
float *Y = ll.output;
printf("---- Y ----\n");
for(j = 0; j < 169; j++) printf("%d: %f\n", j, Y[j]);
printf("\n");
```

#### Reorg Problem
There seems the problem in darknet

#### get_region_boxes speed up
detect.py cfg/yolo.cfg yolo.weight data/dog.jpg
- slow : 0.145544 
- fast : 0.050640
- faster: 0.009280

train.py
- slow: 380ms
- fast: 114ms
- faster: 22ms (batch=64 1.5s)
- fasterer: gpu to cpu  (batch=64 0.15s)
