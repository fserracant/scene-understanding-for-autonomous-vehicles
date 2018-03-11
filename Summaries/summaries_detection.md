# Summaries of image detection papers

## YOLO
[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

YOLO was the first deep network that got rid of region proposal on the object detection pipeline. A single CNN based on GoogLeNet predicts classes and bounding boxes simultaneously in the same pipeline, hence its name "You Only Look Once". Its architecture has the entire image as a input, hence adding contextual information to any bounding box. It is particularly good at discriminating backgrounds, therefore the number of backgrounds detected as objects (false positive rate) is rather small in comparison to Faster R-CNN. One of the main features of this deep network is that it can run at real time (45fps on standard version) due to the fact that the image is passed just one time through the one-line pipeline.

Each image is split into an *SxS* grid and each cell predicts *B*  bounding boxes and *C* confidence probabilities (i.e.: the probability  of an object from class *i* conditioned by the probability of detecting an object in that cell). This procedure introduces more classification mistakes with small objects than the state-of-the-art architectures. This issue is partially addressed by applying regularization in the loss function, increasing the loss weights for the bounding boxes and decreasing it from confidence scores.

The training is performed using a pre-trained net based on *ImageNet*. Several techniques are used to ease the training: the use of leaky *ReLU* activation function, *MSE* loss function, dropout, decreasing manually the learning rate momentum and decay along epochs. Once the network has been trained, inference is done by passing the image through the net and applying non-maximal suppression to merge duplicate detections, since small objects are predicted on one cell and bigger ones on multiple ones.

[YOLO Scheme](Yolo.png)

## SSD
[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)

SSD (Single Shot MultiBox Detector) is an architecture that, like *YOLO*, eliminates the need of an object proposal algorithm as a previous step for object detection. It is faster (59 fps on prediction) and better in accuracy than the *YOLOv1* (however it is not better than the newer *YOLO9000*). The main feature is that it is capable to perform multi-scale predictions. This is achieved by performing prediction on different feature maps with different receptive fields.

The main architecture of SSD is based on the feed-forward convolutional network *VGG* with several convolutional layers at the end, allowing the multi-scale detection. However, this procedure needs a ground truth adaptation. To eliminate the need of object proposals a set of default bouding boxes is fixed to each feature map cell. So the ground truth needs to be expressed as one of this bounding box.

Because of this adaptation, the training procedure is quite tricky and care must be taken to train this network. In order to balance the ratio between positive and negative training examples, *Hard Negative Mining by selection* (limiting the negative-positive ratio to 3:1) has been used during the training stage . The authors also proposed data augmentation, getting patches near the bounding boxes, to improve the accuracy in small objects. This architecture achieved an accuracy similar to *Faster R-CNN* and the speed of *YOLO*'s architecture.

[SSD architecture compared to YOLO](SSD.png)
