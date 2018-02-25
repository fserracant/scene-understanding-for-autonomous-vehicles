# Summaries of image classification papers

## VGG
[Very Deep Convolutional Networks for Large-scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis tempor sit
amet metus eget vestibulum. Quisque sit amet enim non odio ullamcorper
cursus. Mauris fermentum ultrices tortor, eu condimentum elit tempor et.
Etiam mi lacus, imperdiet vitae tristique vel, volutpat sit amet dolor.
Morbi eros ante, accumsan vitae ante vitae, efficitur posuere arcu. Nunc
vitae rutrum justo. Sed eget felis iaculis, sodales dui at, tempus
nulla. Curabitur nunc ligula, sodales quis aliquam non, auctor eget
nisl. Praesent tincidunt finibus mauris accumsan maximus.

[VGG16 Architecture](vgg16.png)
<!-- [<img src="vgg16.png" alt="VGG16 architecture" height="400">](vgg16.png) -->

## ResNet
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis tempor sit
amet metus eget vestibulum. Quisque sit amet enim non odio ullamcorper
cursus. Mauris fermentum ultrices tortor, eu condimentum elit tempor et.
Etiam mi lacus, imperdiet vitae tristique vel, volutpat sit amet dolor.
Morbi eros ante, accumsan vitae ante vitae, efficitur posuere arcu. Nunc
vitae rutrum justo. Sed eget felis iaculis, sodales dui at, tempus
nulla. Curabitur nunc ligula, sodales quis aliquam non, auctor eget
nisl. Praesent tincidunt finibus mauris accumsan maximus.

[ResNet50 architecture](resnet50.png)

## SEnet
[Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)

SEnet introduces the concept of Squeeze and Excitation blocks which builds on top of traditional convolutional ones and significantly improves their performance at nearly no cost. The main idea is to add a new level of flexibility to traditional convolutional layers by enabling the dynamic adaptation of the weights of each feature map through the use of few parameters (1% more than conventional blocks) with the aim to give more importance to relevant features and less to irrelevant ones.

Particularly, a SEnet block (visually in figure a)) takes as input a 3D tensor (width x height x num_channels) and computes a global understanding of each channel by using a global average pooling through the spatial dimensions (width x height), obtaining a 1 x 1 x num_channels output vector. This vector is then passed through a first fully-connected layer with a RELU activation function to add the necessary non-linearities and its complexity (number of output channels) is reduced by a ratio (the authors employed ratio=16), reducing the overall number of parameters. Next, a second FC layer with a sigmoid activation gives a smooth gating function (bounded) which encodes the relative importance of each channel. Finally, each original feature map is multiplied by the output of the “side” network (SEnet module) to obtain the SEnet block’s output.

Thanks to its simplicity, SEnet blocks can be added to any traditional CNN by just replacing the conventional convolutional block by its SEnet equivalent. In the paper (see figure b) below), the authors evaluated ResNet50 with SE blocks (SE-ResNet50) and obtained a similar performance to ResNet101 which has twice as many parameters.
<p align="center"><img src="SEnet_block.png" width="560" height="130"></p>
<p align="center"><i> a) SEnet block </i></p>
<p align="center"><img src="ResNetmod_SEnetmod.png" width="299" height="295"></p>
<p align="center"> <i> b) Module comparison: ResNet vs SE-ResNet equivalent </i> </p>
