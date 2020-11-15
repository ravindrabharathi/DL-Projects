# Architectural Basics : build a CNN with less than 15000 parameters to achieve a validation accuracy of 99.4 or more for MNIST dataset.


The target was to build a Convolutional Neural Network with less than 15000 parameters to achieve a validation accuracy of 99.4 or more for MNIST dataset. The low parameter count becomes important when deploying the model in memory constrained devices used in edge computing . MNIST is one of the more popular (and simpler) datasets to begin your journey in Vision based Deep learning. We will use this MNIST dataset for this exercise.

We built the model step by step . 

**1. Decide on the basic architecture for the network**

In version 1 of the model we tried to get the basic architecture correct. We focused on adding the proper convolution layers that gives a good enough accuracy to start with . We did not bother about the parameter constraints at this point.

In our architecture for every convolution block we start from a small number and increase the number of filters for every additional layer in the block . Typically for a 400x400 image we would go upto a receptive field of 11x11 starting from 32 kernels increasing to 512 kernels such that the network learns enough about edges, gradients ,textures, patterns, parts of object , object and scene in each progressive block . In our case the images are small and relative much simpler and so we will use lesser number of blocks and also lesser layers within each block.

Considering the size of the image (28x28) and the relative simple nature of the classes (digits 0-9) to be classified, we added only two convolution blocks containing 3x3 filters until we got to a channel size of  7x7 .  At the point where the channel size is 7X7 , we added a convolution block of 10 filters of size 7x7 which feeds into a flatten layer followed by softmax activation . Between these two convolution blocks there is a transition block of 1x1 convolution and maxpooling .

We did not add convolution layers after the channel size reached 7x7 . This is because when a 3x3 filter convolves over an image of size 5x5 it sees the pixels a disproportionate number of times as seen below and convovling further doesn't retain enough spatial information   


<img src="https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/pixels-55.png" alt="drawing" width="300"/> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/pixels-77.png" alt="drawing" width="300"/>

All these convolution layers will use ReLU activation function . Using an activation function introduces non-linearity in the network allowing them to learn complex functions .Without these non-linear activation functions the network will only be a stack of linear functions . ReLU activation is one of the simplest and most popular activations used in CNN . It essentially suppresses -ve values from moving forward giving the network a simple rule for retaining or discarding features that it is learning - work towards making values positive if you want something retained and make the values negative if you want to drop something. 

***First convolution layer :***

The first convolution layer : 32 3x3 kernels 

***First Convolution Block :***

The first block allows the network to learn edges and gradients. This block has 2 layers of convolution filters of size 3x3 . The number of kernels in the first layer is 64 (increses from 32 used in the first convolution layer ) and the second layer has 128 kernels . With a global receptive field of 7x7 , the network should be able to learn enough about edges and gradients . 

***Transition Block :*** 
since we want to start the next convolution block at a smaller number of kernels , we used a 1x1 convolution layer to reduce the number of channels after the first convolution block . 1x1 kernel convolution is an effective  way of combining a large number of channels to form a set of smaller number of channels. 
We spatially downsample the channels by using maxpooling of size 2x2. Maxpooling of size 2x2 will reduce the channel size by half while doubling the global receptive field 

***Second convolution block :*** 

The second block will allow the network to form the important parts that make up the digits . This contains two layers of 3x3 kernels . Firts layer will have 64 kernels and the second will have 128 kernels . 

1x1 convolution to transition to 10 channels :

After the second convolution block we add a convolution layer of 10 1x1 kernels .  Since we have only 10 classes , we combine the 16 channels from earlier layers to form 10 channels .

***Last Layer : 7x7 kernel*** 

Since we stopped our second convolution block at an activation size of 7x7 , we add a convolution layer of 10 numbers of 7x7 kernels as the last layer in order to send a 1x1x10 output to the Flatten layer .
It is important not to have ReLU activation for this 1x1 layer since we want all values from the convolution to go to the Softmax activation to make its prediction . If we use a ReLu activation , the -ve values will be suppressed and the network will be unable to train in an optimal manner.

***Flatten:*** 

These 10 channel outputs are fed to a Flatten layer that converts the 2d array representation to a 1d shape . 

***Softmax activation:*** 

A softmax activation layer at the end outputs the class probabilities of these 10 classes which in our case are the digits 0 to 9 . 

**Result : First iteration of the model with 195k parameters and achieved a max validation accuracy of 99.22 within 10 epochs**

**2. Fine tune parameters to comply with the 15000 limit**

In version 2 of the model we got the number of parameters down below the 15k limit

We reduced the number of kernels in the following manner

***First convolution layer :***

The first convolution layer will have 10 numbers 3x3 kernels instead of 32

***First Convolution Block :***

The number of kernels in the first layer was changed to 12 instead of 64 and the second layer will now have 16 kernels instead of 128 kernels . With a global receptive field of 7x7 , the network should be able to learn enough about edges and gradients .

***Transition Block :*** since we want to start the next convolution block at a smaller number of kernels , we used a 1x1 convolution layer to reduce the number of channels after the first convolution block . 1x1 kernel convolution is an effective way of combining a large number of channels to form a set of smaller number of channels. We will use 10 numbers of 1x1 kernels to bring the number of channels down to 10. We will also spatially downsample the channels by using maxpooling of size 2x2. Maxpooling of size 2x2 will reduce the channel size by half while doubling the global receptive field

***Second convolution block :***

The second block will allow the network to form the important parts that make up the digits . This will contain two layers of 3x3 kernels . First layer will have 12 kernels (down from 64) and the second will have 16 kernels (down from 128 kernels ) .

No changes in the last few layers 

**Result : We trained the model for 20 epochs and got a max validation accuracy of 99.17. The number of parameters came down to within limits at 10,926** 

**3. Add improvements to the network using Batch Normalization** 

In this iteration we tuned the performance of the model by adding BatchNormalization .

Batch Normalization is a way for the network take care of internal covariate shift in the features and was first introduced in this paper titled Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

We followed how BatchNormalization was used in the paper i.e Convolution followed Batch Normalization and then Activation although recently some practioners prefer using BatchNormalization after activation citing better performance.
We did not try a higher learning rate at this point eventhough it is recommended by the authors of the above paper

**Result/Observation : modelreached a max validation accuracy of 99.41 in the 20th epoch. The training time per epoch increased due to the addition of BatchNormalization. Total params: 11310, Trainable params: 11118 , Non-trainable params: 192**

**4. Improve validation accuracy using Dropouts, tweaking the learning rate and larger batch size for training**

In this iteration we tuned the performance of the model by adding Dropouts.

Basically Dropout helps in regularizing a network against overrfitting by randomly dropping a proportion of the signals while training. The absence of some of the units force the rest of the active units to learn better . The concept of Dropout was first present in this paper titled [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html). 

![dropouts](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/dropouts.png)

We dropped 10% of the signals after each convolution layer (excluding the transition blocks and the last layer) .

We also used a higher learning rate of 0.003 and a simplified learning rate scheduler that decreased lr every 3rd epoch subject to a min of 0.0005 .  

We also increased the Batch size to 128 from 32 since it would be better for the model to see a larger sampling of images per training step 

**Result/Observation : No change in total parameters. The model crossed the 99.4 validation accuracy in the 14th epoch and reached a max validation accuracy of 99.52 in 60 epochs. The gap between training and validation accuracy narrowed and also training accuracy was initially lower than validation accuracy since the dropped units partcipate fully during validation cycle**

**Further Improvements / Points to consider**

1. We did not use GlobalAveragePooling . We could have used this instead of the last convolution layer.  

2. Use a better learning rate scheduler that takes into account whether accuracy is increasing before adjusting learning rate




