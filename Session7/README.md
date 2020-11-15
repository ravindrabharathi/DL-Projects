# Effective Receptive Filed Calculations , ENAS' Discovered Convolution Network for CIFAR 10 

This week we have focussed on two things :

1. How to calculate the effective or global receptive field at any layer 
2. Design a Network as per ENAS' discovered Convolution Network for CIFAR 10 dataset 





### Effective Receptive Field calculations 

The formula for RF calculation is as shown below 


![Effective RF calc](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/rf_calc.png)

Using above formula we will calculate the Effective RF for the GoogleNet architecture 

![GoogleNet](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/googleNet.png)




| Layer | Kernel size(k) | stride(s) |jump(j) | Effective Receptive Field |Comments |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Input  |  |  | 1 | 1 |
| convolution 7×7/2  | 7 | 2 | 2 | 7 |
| max pool 3×3/2     |  3 | 2 | 4 | 11 |
| convolution 3×3/1  |  3 | 1 | 4 | 19 |
| max pool 3×3/2  |  3 | 2 | 8 | 27 |
| inception (3a)  |  5| 1 | 8 | 59 | inception has a 1x1, 3x3 , 5x5 and a max pool 3x3 . all have stride=1.We will use the 5x5 kernel group for this calculation |
| inception (3b)  |  5 | 1 | 8 | 91 |
| max pool 3×3/2  |  3 | 2 | 16 | 107 |
| inception (4a)  |  5 | 1 | 16 | 171 |
| inception (4b)  |  5 | 1 | 16 | 235 |
| inception (4c)  |  5 | 1 | 16 | 299 |
| inception (4d)  |  5 | 1 | 16 | 363 |
| inception (4e)  |  5 | 1 | 16 | 427 |
| max pool 3×3/2  |  3 | 2 | 32 | 459 |
| inception (5a)  |  5 | 2 | 32 | 587 |
| inception (5b)  |  5 | 2 | 32 | 715 |
| avg pool 7×7/1  |  7 | 2 | 32 | 907 |
| dropout (40%)	  |  
| linear |  
| softmax |


### Design of ENAS' Discovered Convolution Network for CIFAR 10 Dataset 

We will also design a Model that is based on the following ENAS' discovered convolution architecture for CIFAR 10 dataset as presented in the paper ['Efficient Neural Architecture Search via Parameter Sharing'](https://arxiv.org/pdf/1802.03268.pdf)

### The model will be designed as per the following ENAS's discovered network 
![Enas discovered Network](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/enas-network.png)

#### Design considerations from the paper 

We will also consider the following points (quoted verbatim from the paper) mentioned in the paper while designing the network 

>>#### B. Details on CIFAR-10 Experiments
>>We find the following tricks crucial for achieving good performance with ENAS. Standard NAS (Zoph & Le, 2017;
Zoph et al., 2018) rely on these and other tricks as well.
Structure of Convolutional Layers. Each convolution in our model is applied in the order of relu-convbatchnorm (Ioffe & Szegedy, 2015; He et al., 2016b). Additionally, in our micro search space, each depthwise separable convolution is applied twice (Zoph et al., 2018).

>>#### Stabilizing Stochastic Skip Connections. 
>>If a layer receives skip connections from multiple layers before it, then
these layers’ outputs are concatenated in their depth dimension, and then a convolution of filter size 1 × 1 (followed
by a batch normalization layer and a ReLU layer)

>>#### Global Average Pooling 
>>After the final convolutional
layer, we average all the activations of each channel and
then pass them to the Softmax layer. 

>>#### Number of filters (mentioned in section 3.2. Image Classification on CIFAR-10 of the paper under Results (page 5))
>>If we keep
the architecture, but increase the number of filters in the
network’s highest layer to 512, then the test error decreases
to 3.87%, which is not far away from NAS’s best model,
whose test error is 3.65% . 


We built two models :
Model 1 : one with increasing kernels sizes of 32 -> 64 -> 128 -> 256 for the initial blocks and 64->128->256-512 for the last block. This model has 4,300,221 parameters 

Model 2 : increasing kernels sizes of 8 -> 16 -> 32 -> 64 for the initial blocks and 16->32->64-128 for the last block. This Model has 284,301 parameters

Both models used an GlobalAveragePooling Layer before the Softmax activation. BatchNormalization, ImageNormalization and Image Augmentation techniques were used for both models

Model 1 trained for 100 epochs and reached a max validation accuracy of 92.48 .

Model 2 trained for 100 epochs and reached a max validation accuracy of 88.61 .

More training epochs and better augmentation techniques as mentioned in the paper(*standardizing the data, using horizontal flips with 50% probability, zero-padding and random crops, and finally Cutout with 16x16 pixels*) would have yielded better results.


	
#### [Link to Notebook](https://github.com/ravindrabharathi/Project1/blob/master/Session7/Assignment7B.ipynb) 				
					
	
