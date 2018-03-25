# Summaries of image segmentation papers

## FCN
[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

A Fully Convolutional Network (FCN) could be seen not as a network by itself, but as a  modification of other network. The main idea is to transform the given network’s layers into convolutional layers. The original paper shows experiments with AlexNet, GoogLeNet and VGG16 being this latter the one which best performs. 

In more detail, first they “decapitate” the network removing the classification layer. Afterwards all the fully connected layers (FC) are converted to convolutional ones, keeping the number of channels but extending it for every “pixel”. For instance, if a FC has *d*  parameters, namely it is a *1 x d* dimensional array then is converted to a *h x w x d* array. These *h* and *w* are design parameters as any regular conv layer.
Subsequently a conv layer of size *h x w x k*, being *k* the number of classes, is appended to the network. This provides a coarse classification scores. Lastly, an upsampling stage which yields an equally sized output than the input image is appended, turning the coarse prediction into a dense prediction. 
The upsampling is done by fractionally strided convolutions, also called transposed convolutions or informally “deconvolutions”. This technique offers and end-to-end training

Only doing that, the semantic segmentation if quite coarse [Fig. a](#fcn_refinement). Therefore a refinement strategy is proposed to improve spatial precision. The strategy is to append those coarse predictions and upsampling stages on different layers of the network [Fig. b](#fcn_net), taking advantage of the implicit hierarchy of receptive fields in consecutive conv layers. As earlier layers are combined the spatial precision is increased [Fig. a](#fcn_refinement). The authors exposed that using more than 3 layers of predictions is not worthy due to a low further improvement and an increasing complexity.

<p align="center"><img src="fcn_refining.png" width="480"></p>
<a name="fcn_refinement"><p align="center"><i> a) FCN refining prediction map with lower layers </i></p>

<p align="center"><img src="fcn_net.png" width="480"></p>
<a name="fcn_net"><p align="center"><i> b) FCN network </i></p>

The results are quite better than the state-of-the-art of the moment, like SDS. Compared with SDS, the authors reported an inference time is 286 times lower and it has a +20% in mean IU on PASCAL VOC2011/12 datasets. Also, training is 5 times faster than AlexNet on a GPU. They fine-tuned the networks of the experiment to leverage the already trained filters. Neither class balancing nor data augmentation is proposed in the original paper. 

To sum up, the network is able to recover fine structures, to separate closely interacting objects and it is robust to occluders [Fig. c](#fcn_vs_sds).

<p align="center"><img src="fcn_vs_sds.png" width="480"></p>
<a name="fcn_vs_sds"><p align="center"><i> c) Comparison between FCN and SDS </i></p>

An official implementation on Caffe is publicly [available in GitHub](https://github.com/shelhamer/fcn.berkeleyvision.org)

## SegNet
[A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)

