# Summaries of image detection papers

## YOLO
[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

YOLO was the first deep network that got rid of region proposal on the object detection pipeline. A single CNN based on GoogLeNet predicts the classes and bounding boxes simultaneously, in the same pipeline, hence its name "You Only Look Once". This architecture works with the entire image adding contextual information to the object detections. It is particularly good at discriminating background, the number of background detected as objects (false positive rate) is small. One of the main features of this deep network is that it can run at real time (45fps on standard version) due to the fact that the image is passed just one time through a CNN.

Each image is split into an S x S grid and each cell predicts B bounding boxes and C confidence probabilities (i.e.: the probability that an object is from class i if there is an object in that cell). This procedure introduce more classification mistakes with small objects than state of the art architectures. It is partially addressed by applying regularization (increase loss for the bounding boxes and decrease loss from confidence scores).

The network training is performed using a pretrained net based on ImageNet. Several techniques are used to ease the training: use of leaky relu activation function, MSE loss function, dropout, learning rate momentum and decay, it is also decreased along epochs. Once the network has been trained, inference is done by passing the image through the net and applying non-maximal suppression to merge duplicate detections, small objects are detected on one cell and bigger ones occupy more than one cell.

[YOLO Scheme](Yolo.png)

## SSD
[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)

SSD (Single Shot MultiBox Detector) is another architecture that, like YOLO, eliminates the need of an object proposal algorithm on the object detection pipeline. It is faster and better then standard YOLO (however it is not better than newer YOLO9000), it can process images at real time (59fps). The main feature of this net is that it is capable to perform multiscale predictions. This is achieved by performing prediction on different feature maps with different receptive fields.

The main architecture of SSD is based on the feed-forward convolutional network VGG with several convolutional layers at the end to allow the detection of multiple scales objects. However, this procedure needs a ground truth adaptation. A set of default bounding boxes are associated to a feature map cell. The features predict the detection offset to the default bounding box and the probability that it contains an object of certain class.

Due to the fact that the object ground truth must be adapted, the training procedure is quite tricky and care must be taken to train this network. Hard negative mining has been used during the training stage in order to balance the ratio between positive and negative training examples. The authors also proposed data augmentation for improving the accuracy of small objects. This architecture achieved accuracy similar to Faster R-CNN however with the speed of the YOLO architecture.

[SSD architecture compared to YOLO](SSD.png)
