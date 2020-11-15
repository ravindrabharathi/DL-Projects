# Visualize where the Convolutional Neural Network is looking at using Grad-CAM for CIFAR10 dataset

When solving image classification problems , it would be helpful for us to know what information from the image is being used by the network to make its preditions. Grad-CAM is a way for us to visualize the pixels in the activation channels that contribute most actively to a certain prediction . This will help us to fine tune the model in such a way that it uses all/most of the information belonging to the object being classified while making a prediction as opposed to using only small parts of the object or on the fringes or background surrounding the object being classified. This will help the model learn more about the features of a particular class .

In this exercise we will do a Grad-CAM visualization on 10 misclassified images of CIFAR10 dataset. We will use a ResNet18 model based on the notes for CIFAR-10 in the [RESNET paper](https://arxiv.org/pdf/1512.03385.pdf) 

Based on these experiments of the authors on CIFAR-10 a Resnet20 Model has been defined in [https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) . We will use ResNetv2 from this project  for CIFAR-10 dataset . 

We will make a few small changes:
1. Remove the last Dense layer 
2. Add a Conv2D 1x1 to reduce the number of channels to 10 (reducer1)

The modified ResNet20 Model definition file is at 
[https://raw.githubusercontent.com/srbharathee/DL/master/cifar10_resnet20.py](https://raw.githubusercontent.com/srbharathee/DL/master/cifar10_resnet20.py)

### Notebook link : [https://github.com/ravindrabharathi/Project1/blob/master/EIP4/session4/002_EIP4_CIFAR10_ResNet_Grad_CAM.ipynb](https://github.com/ravindrabharathi/Project1/blob/master/EIP4/session4/002_EIP4_CIFAR10_ResNet_Grad_CAM.ipynb)

### Model training logs

```
Epoch 1/50
epoch  1 : setting learning rate to  0.01
391/391 [==============================] - 39s 101ms/step - loss: 2.2419 - acc: 0.3350 - val_loss: 2.0195 - val_acc: 0.4124

Epoch 00001: val_acc improved from -inf to 0.41240, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 2/50
epoch  2 : setting learning rate to  0.04166666666666667
391/391 [==============================] - 34s 87ms/step - loss: 1.9183 - acc: 0.4521 - val_loss: 2.1830 - val_acc: 0.3985

Epoch 00002: val_acc did not improve from 0.41240
Epoch 3/50
epoch  3 : setting learning rate to  0.07333333333333333
391/391 [==============================] - 34s 87ms/step - loss: 1.6503 - acc: 0.5499 - val_loss: 1.6953 - val_acc: 0.5468

Epoch 00003: val_acc improved from 0.41240 to 0.54680, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 4/50
epoch  4 : setting learning rate to  0.105
391/391 [==============================] - 34s 87ms/step - loss: 1.4477 - acc: 0.6111 - val_loss: 1.5367 - val_acc: 0.5881

Epoch 00004: val_acc improved from 0.54680 to 0.58810, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 5/50
epoch  5 : setting learning rate to  0.1366666666666667
391/391 [==============================] - 34s 87ms/step - loss: 1.3181 - acc: 0.6494 - val_loss: 1.3181 - val_acc: 0.6555

Epoch 00005: val_acc improved from 0.58810 to 0.65550, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 6/50
epoch  6 : setting learning rate to  0.16833333333333336
391/391 [==============================] - 34s 88ms/step - loss: 1.2085 - acc: 0.6812 - val_loss: 1.6957 - val_acc: 0.5817

Epoch 00006: val_acc did not improve from 0.65550
Epoch 7/50
epoch  7 : setting learning rate to  0.2
391/391 [==============================] - 34s 87ms/step - loss: 1.1304 - acc: 0.7008 - val_loss: 1.3823 - val_acc: 0.6333

Epoch 00007: val_acc did not improve from 0.65550
Epoch 8/50
epoch  8 : setting learning rate to  0.19545454545454546
391/391 [==============================] - 34s 87ms/step - loss: 1.0286 - acc: 0.7319 - val_loss: 1.1516 - val_acc: 0.6813

Epoch 00008: val_acc improved from 0.65550 to 0.68130, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 9/50
epoch  9 : setting learning rate to  0.19090909090909092
391/391 [==============================] - 34s 87ms/step - loss: 0.9742 - acc: 0.7498 - val_loss: 1.3758 - val_acc: 0.6457

Epoch 00009: val_acc did not improve from 0.68130
Epoch 10/50
epoch  10 : setting learning rate to  0.18636363636363637
391/391 [==============================] - 34s 87ms/step - loss: 0.9317 - acc: 0.7606 - val_loss: 1.0343 - val_acc: 0.7489

Epoch 00010: val_acc improved from 0.68130 to 0.74890, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 11/50
epoch  11 : setting learning rate to  0.18181818181818182
391/391 [==============================] - 34s 87ms/step - loss: 0.8873 - acc: 0.7712 - val_loss: 1.2381 - val_acc: 0.6689

Epoch 00011: val_acc did not improve from 0.74890
Epoch 12/50
epoch  12 : setting learning rate to  0.17727272727272728
391/391 [==============================] - 34s 87ms/step - loss: 0.8646 - acc: 0.7825 - val_loss: 2.4906 - val_acc: 0.4788

Epoch 00012: val_acc did not improve from 0.74890
Epoch 13/50
epoch  13 : setting learning rate to  0.17272727272727273
391/391 [==============================] - 34s 87ms/step - loss: 0.8358 - acc: 0.7888 - val_loss: 1.1097 - val_acc: 0.7205

Epoch 00013: val_acc did not improve from 0.74890
Epoch 14/50
epoch  14 : setting learning rate to  0.16818181818181818
391/391 [==============================] - 34s 87ms/step - loss: 0.8133 - acc: 0.7967 - val_loss: 0.9751 - val_acc: 0.7420

Epoch 00014: val_acc did not improve from 0.74890
Epoch 15/50
epoch  15 : setting learning rate to  0.16363636363636364
391/391 [==============================] - 34s 87ms/step - loss: 0.7923 - acc: 0.8023 - val_loss: 0.9081 - val_acc: 0.7608

Epoch 00015: val_acc improved from 0.74890 to 0.76080, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 16/50
epoch  16 : setting learning rate to  0.1590909090909091
391/391 [==============================] - 34s 87ms/step - loss: 0.7775 - acc: 0.8058 - val_loss: 1.1588 - val_acc: 0.7034

Epoch 00016: val_acc did not improve from 0.76080
Epoch 17/50
epoch  17 : setting learning rate to  0.15454545454545454
391/391 [==============================] - 34s 87ms/step - loss: 0.7634 - acc: 0.8122 - val_loss: 1.4047 - val_acc: 0.6699

Epoch 00017: val_acc did not improve from 0.76080
Epoch 18/50
epoch  18 : setting learning rate to  0.15
391/391 [==============================] - 34s 87ms/step - loss: 0.7509 - acc: 0.8159 - val_loss: 1.0373 - val_acc: 0.7364

Epoch 00018: val_acc did not improve from 0.76080
Epoch 19/50
epoch  19 : setting learning rate to  0.14545454545454545
391/391 [==============================] - 34s 87ms/step - loss: 0.7388 - acc: 0.8208 - val_loss: 1.3086 - val_acc: 0.6706

Epoch 00019: val_acc did not improve from 0.76080
Epoch 20/50
epoch  20 : setting learning rate to  0.14090909090909093
391/391 [==============================] - 34s 87ms/step - loss: 0.7197 - acc: 0.8248 - val_loss: 1.0509 - val_acc: 0.7221

Epoch 00020: val_acc did not improve from 0.76080
Epoch 21/50
epoch  21 : setting learning rate to  0.13636363636363635
391/391 [==============================] - 34s 87ms/step - loss: 0.7150 - acc: 0.8285 - val_loss: 0.8462 - val_acc: 0.7894

Epoch 00021: val_acc improved from 0.76080 to 0.78940, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 22/50
epoch  22 : setting learning rate to  0.13181818181818183
391/391 [==============================] - 34s 87ms/step - loss: 0.6966 - acc: 0.8350 - val_loss: 0.8752 - val_acc: 0.7856

Epoch 00022: val_acc did not improve from 0.78940
Epoch 23/50
epoch  23 : setting learning rate to  0.12727272727272726
391/391 [==============================] - 34s 87ms/step - loss: 0.6822 - acc: 0.8377 - val_loss: 0.8339 - val_acc: 0.7841

Epoch 00023: val_acc did not improve from 0.78940
Epoch 24/50
epoch  24 : setting learning rate to  0.12272727272727273
391/391 [==============================] - 34s 87ms/step - loss: 0.6718 - acc: 0.8408 - val_loss: 0.9354 - val_acc: 0.7497

Epoch 00024: val_acc did not improve from 0.78940
Epoch 25/50
epoch  25 : setting learning rate to  0.11818181818181818
391/391 [==============================] - 34s 87ms/step - loss: 0.6604 - acc: 0.8454 - val_loss: 0.8739 - val_acc: 0.7828

Epoch 00025: val_acc did not improve from 0.78940
Epoch 26/50
epoch  26 : setting learning rate to  0.11363636363636363
391/391 [==============================] - 34s 87ms/step - loss: 0.6495 - acc: 0.8478 - val_loss: 0.7783 - val_acc: 0.8154

Epoch 00026: val_acc improved from 0.78940 to 0.81540, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 27/50
epoch  27 : setting learning rate to  0.10909090909090909
391/391 [==============================] - 34s 87ms/step - loss: 0.6399 - acc: 0.8509 - val_loss: 0.7881 - val_acc: 0.8079

Epoch 00027: val_acc did not improve from 0.81540
Epoch 28/50
epoch  28 : setting learning rate to  0.10454545454545454
391/391 [==============================] - 34s 88ms/step - loss: 0.6262 - acc: 0.8550 - val_loss: 1.0268 - val_acc: 0.7627

Epoch 00028: val_acc did not improve from 0.81540
Epoch 29/50
epoch  29 : setting learning rate to  0.09999999999999999
391/391 [==============================] - 34s 88ms/step - loss: 0.6142 - acc: 0.8593 - val_loss: 0.7899 - val_acc: 0.8186

Epoch 00029: val_acc improved from 0.81540 to 0.81860, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 30/50
epoch  30 : setting learning rate to  0.09545454545454544
391/391 [==============================] - 34s 87ms/step - loss: 0.6041 - acc: 0.8631 - val_loss: 0.6752 - val_acc: 0.8413

Epoch 00030: val_acc improved from 0.81860 to 0.84130, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 31/50
epoch  31 : setting learning rate to  0.09090909090909091
391/391 [==============================] - 34s 88ms/step - loss: 0.5892 - acc: 0.8663 - val_loss: 0.7773 - val_acc: 0.8198

Epoch 00031: val_acc did not improve from 0.84130
Epoch 32/50
epoch  32 : setting learning rate to  0.08636363636363636
391/391 [==============================] - 34s 87ms/step - loss: 0.5796 - acc: 0.8700 - val_loss: 0.7597 - val_acc: 0.8161

Epoch 00032: val_acc did not improve from 0.84130
Epoch 33/50
epoch  33 : setting learning rate to  0.08181818181818182
391/391 [==============================] - 34s 88ms/step - loss: 0.5663 - acc: 0.8737 - val_loss: 0.8280 - val_acc: 0.8016

Epoch 00033: val_acc did not improve from 0.84130
Epoch 34/50
epoch  34 : setting learning rate to  0.07727272727272727
391/391 [==============================] - 35s 88ms/step - loss: 0.5556 - acc: 0.8760 - val_loss: 0.7974 - val_acc: 0.8033

Epoch 00034: val_acc did not improve from 0.84130
Epoch 35/50
epoch  35 : setting learning rate to  0.07272727272727272
391/391 [==============================] - 35s 89ms/step - loss: 0.5439 - acc: 0.8784 - val_loss: 0.8423 - val_acc: 0.7890

Epoch 00035: val_acc did not improve from 0.84130
Epoch 36/50
epoch  36 : setting learning rate to  0.06818181818181818
391/391 [==============================] - 35s 89ms/step - loss: 0.5293 - acc: 0.8822 - val_loss: 0.7696 - val_acc: 0.8206

Epoch 00036: val_acc did not improve from 0.84130
Epoch 37/50
epoch  37 : setting learning rate to  0.06363636363636363
391/391 [==============================] - 35s 88ms/step - loss: 0.5144 - acc: 0.8881 - val_loss: 0.7073 - val_acc: 0.8405

Epoch 00037: val_acc did not improve from 0.84130
Epoch 38/50
epoch  38 : setting learning rate to  0.05909090909090908
391/391 [==============================] - 34s 87ms/step - loss: 0.5019 - acc: 0.8908 - val_loss: 0.6955 - val_acc: 0.8331

Epoch 00038: val_acc did not improve from 0.84130
Epoch 39/50
epoch  39 : setting learning rate to  0.054545454545454536
391/391 [==============================] - 34s 88ms/step - loss: 0.4894 - acc: 0.8942 - val_loss: 0.6049 - val_acc: 0.8648

Epoch 00039: val_acc improved from 0.84130 to 0.86480, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 40/50
epoch  40 : setting learning rate to  0.04999999999999999
391/391 [==============================] - 34s 88ms/step - loss: 0.4711 - acc: 0.8994 - val_loss: 0.6950 - val_acc: 0.8339

Epoch 00040: val_acc did not improve from 0.86480
Epoch 41/50
epoch  41 : setting learning rate to  0.04545454545454544
391/391 [==============================] - 34s 88ms/step - loss: 0.4536 - acc: 0.9039 - val_loss: 0.6147 - val_acc: 0.8641

Epoch 00041: val_acc did not improve from 0.86480
Epoch 42/50
epoch  42 : setting learning rate to  0.040909090909090895
391/391 [==============================] - 34s 88ms/step - loss: 0.4388 - acc: 0.9076 - val_loss: 0.6778 - val_acc: 0.8389

Epoch 00042: val_acc did not improve from 0.86480
Epoch 43/50
epoch  43 : setting learning rate to  0.03636363636363635
391/391 [==============================] - 34s 87ms/step - loss: 0.4236 - acc: 0.9135 - val_loss: 0.5941 - val_acc: 0.8708

Epoch 00043: val_acc improved from 0.86480 to 0.87080, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 44/50
epoch  44 : setting learning rate to  0.0318181818181818
391/391 [==============================] - 34s 88ms/step - loss: 0.4014 - acc: 0.9195 - val_loss: 0.6007 - val_acc: 0.8618

Epoch 00044: val_acc did not improve from 0.87080
Epoch 45/50
epoch  45 : setting learning rate to  0.027272727272727254
391/391 [==============================] - 34s 88ms/step - loss: 0.3854 - acc: 0.9244 - val_loss: 0.5154 - val_acc: 0.8873

Epoch 00045: val_acc improved from 0.87080 to 0.88730, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 46/50
epoch  46 : setting learning rate to  0.022727272727272707
391/391 [==============================] - 34s 88ms/step - loss: 0.3665 - acc: 0.9286 - val_loss: 0.4828 - val_acc: 0.8947

Epoch 00046: val_acc improved from 0.88730 to 0.89470, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 47/50
epoch  47 : setting learning rate to  0.01818181818181816
391/391 [==============================] - 34s 87ms/step - loss: 0.3462 - acc: 0.9373 - val_loss: 0.4968 - val_acc: 0.8955

Epoch 00047: val_acc improved from 0.89470 to 0.89550, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 48/50
epoch  48 : setting learning rate to  0.013636363636363613
391/391 [==============================] - 34s 88ms/step - loss: 0.3321 - acc: 0.9403 - val_loss: 0.4384 - val_acc: 0.9070

Epoch 00048: val_acc improved from 0.89550 to 0.90700, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 49/50
epoch  49 : setting learning rate to  0.009090909090909066
391/391 [==============================] - 34s 88ms/step - loss: 0.3072 - acc: 0.9474 - val_loss: 0.4253 - val_acc: 0.9139

Epoch 00049: val_acc improved from 0.90700 to 0.91390, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5
Epoch 50/50
epoch  50 : setting learning rate to  0.004545454545454519
391/391 [==============================] - 34s 88ms/step - loss: 0.2947 - acc: 0.9514 - val_loss: 0.4202 - val_acc: 0.9176

Epoch 00050: val_acc improved from 0.91390 to 0.91760, saving model to /gdrive/My Drive/EIP4/session4/best_model.h5

```

### model accuracy after 50 epochs was 91.76 with still capacity to train further as training accuracy was only at 95.14 

### GRADCAM results 
We plotted GRADCAM heatmaps for 10 mis-classified images from the test set 

![GradCam-results](https://raw.githubusercontent.com/ravindrabharathi/Grad-cam/master/gradcam.png)



