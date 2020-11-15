# IMAGE CLASSIFICATION FOR CIFAR 10 DATASET 

In this section we trained a convolutional Neural Network to classify images in the CIFAR 10 Dataset 

There are two notebooks : 

### [Assignment 6A](https://github.com/ravindrabharathi/Project1/blob/master/Session6/Assignment_6A.ipynb) 
Here we first trained a network which used Dense layers and a large number of parameters (1.17 Million) to acquire a base accuracy and then corrected the network by doing the following ;
1. Removed the Dense layers 
2. Modified architecture to use convolution blocks of layers with increasing kernels (like a flat topped pyramid), introduced transition blocks (of a pointwise convolution followed by Max pooling) between these convolution blocks 
3. Reduced the number of kernels (and thus the number of params and tendency to ovrfit) ,
4. Added BatchNormalization to reduce the effect of internal covariate shift, 
5. Placed Dropouts for Regularization at appropriate places , reduced the number of dropouts and the % too , 
6. Added Image Augmentation to help the newtork generalize better / avoid overfitting , 
7. Added a simple learning rate scheduler with a higher initial learning rate for faster convergence during training. 

We trained the new Model for 100 epochs and achieved a training accuracy (94%) higher than the base accuracy and also observed that the gap between the training accuracy and validation accuracy had narrowed.



### [Assignment 6B](https://github.com/ravindrabharathi/Project1/blob/master/Session6/Assignment_6B.ipynb)

In this section we used the new Model from the previous exercise and modified the network to include the following 

1. A normal convolution using 3x3 kernel 
2. Spatial separable convolution 3x1 followed by 1x3 
3. Depthwise separable convolution using Keras' SeparableConv2D of kernel size 3 
4. Grouped convolution containing filter groups of 3x3 and 5x5 
5. Grouped convolution containing filter groups of 3x3 with dilation rate 1 (no dilation) and another 3x3 with dilation rate of 2 ( a gap of one cell separating each of the kernel weights) 

We retained all the previous improvements like BatchNormalization, dropouts, absence of dense layers , image augmentations , learing rate scheduler in the network and trained it for 50 epochs. A max training accuracy of 92.27 and validation accuracy of 85.48 was achieved.  
