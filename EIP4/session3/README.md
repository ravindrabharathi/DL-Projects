# Train a CNN model built with only Depthwise Separable Convolutions to classify CIFAR-10 image samples


### Base Model accuracy

Accuracy on test data is: 82.81

Base model accuracy at 50 epochs is 82.81 and max accuracy is 83.01

### Model definition with output channel size and RF values as comments after each layer (this model has 1x1 bottleneck layers before max pooling to reduce number of parameters) 
```
# Define the model
model = Sequential()
#model.add(Dropout(0.1 , input_shape=(32,32,3)))
model.add(SeparableConv2D(64, 3, 3, border_mode='same',use_bias=False,depthwise_initializer=wt_init,pointwise_initializer=wt_init, input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

# output channel size = 32 x 32 , RF =3

model.add(SeparableConv2D(96, 3, 3,use_bias=False,depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

#output channel size = 30 x 30 , RF =5 


model.add(SeparableConv2D(96, 3, 3,use_bias=False,depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

# output channel size = 28 x 28, RF =7

#bottleneck 
model.add(Convolution2D(64,1,1,use_bias=False,kernel_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))

# output channel size = 28 x 28 , RF =7

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

# output channel size = 14 x 14 , RF = 8

model.add(SeparableConv2D(64, 3, 3, use_bias=False,border_mode='same',depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.03))

# output channel size = 14 x 14 , RF =12

model.add(SeparableConv2D(96, 3, 3,use_bias=False,border_mode='same',depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.03))

# output channel size = 14 x 14 , RF = 16

model.add(SeparableConv2D(192, 3, 3,use_bias=False,border_mode='same',depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

# output channel size = 14 x 14 , RF = 20

#bottleneck 
model.add(Convolution2D(64,1,1,use_bias=False,kernel_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))

# output channel size = 14 x 14 , RF = 20

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

# output channel size = 7 x 7 , RF = 22


model.add(SeparableConv2D(96, 3, 3,use_bias=False, border_mode='same',depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

# output channel size = 7 x 7 , RF =30 

model.add(SeparableConv2D(192, 3, 3,use_bias=False,depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))

# output channel size = 5 x 5 , RF =38 

model.add(Convolution2D(10,1,1,use_bias=False,kernel_initializer=wt_init))

# output channel size = 5 x 5 , RF = 38

model.add(AveragePooling2D(5,5))

# output channel size = 1 x 1 , RF = 54 

model.add(Flatten())
model.add(Activation('softmax'))


```

### Model parameters 

> Total params: 98,171

> Trainable params: 96,123

> Non-trainable params: 2,048

### Model training log (Model with 1x1 bottlnecks)
```
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.02.
391/391 [==============================] - 76s 193ms/step - loss: 1.6532 - acc: 0.3822 - val_loss: 4.8997 - val_acc: 0.2260
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0151630023.
391/391 [==============================] - 62s 159ms/step - loss: 1.1477 - acc: 0.5864 - val_loss: 1.9773 - val_acc: 0.4981
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0122100122.
391/391 [==============================] - 63s 160ms/step - loss: 0.9435 - acc: 0.6632 - val_loss: 1.1069 - val_acc: 0.6409
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0102197241.
391/391 [==============================] - 63s 161ms/step - loss: 0.8012 - acc: 0.7177 - val_loss: 0.8471 - val_acc: 0.7145
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0087873462.
391/391 [==============================] - 62s 159ms/step - loss: 0.7095 - acc: 0.7511 - val_loss: 0.7303 - val_acc: 0.7502
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0077071291.
391/391 [==============================] - 62s 159ms/step - loss: 0.6417 - acc: 0.7757 - val_loss: 0.8689 - val_acc: 0.7166
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.006863418.
391/391 [==============================] - 62s 159ms/step - loss: 0.5888 - acc: 0.7921 - val_loss: 0.7993 - val_acc: 0.7304
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0061862048.
391/391 [==============================] - 62s 159ms/step - loss: 0.5467 - acc: 0.8094 - val_loss: 0.6488 - val_acc: 0.7790
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0056306306.
391/391 [==============================] - 62s 159ms/step - loss: 0.5107 - acc: 0.8214 - val_loss: 0.5758 - val_acc: 0.8066
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0051666236.
391/391 [==============================] - 62s 159ms/step - loss: 0.4753 - acc: 0.8346 - val_loss: 0.6025 - val_acc: 0.7972
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0047732697.
391/391 [==============================] - 62s 159ms/step - loss: 0.4454 - acc: 0.8439 - val_loss: 0.5475 - val_acc: 0.8171
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0044355733.
391/391 [==============================] - 62s 159ms/step - loss: 0.4254 - acc: 0.8508 - val_loss: 0.5697 - val_acc: 0.8109
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0041425021.
391/391 [==============================] - 62s 159ms/step - loss: 0.4055 - acc: 0.8582 - val_loss: 0.5413 - val_acc: 0.8125
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0038857587.
391/391 [==============================] - 62s 159ms/step - loss: 0.3846 - acc: 0.8635 - val_loss: 0.5435 - val_acc: 0.8189
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0036589828.
391/391 [==============================] - 62s 159ms/step - loss: 0.3650 - acc: 0.8730 - val_loss: 0.5118 - val_acc: 0.8287
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0034572169.
391/391 [==============================] - 62s 159ms/step - loss: 0.3439 - acc: 0.8808 - val_loss: 0.5163 - val_acc: 0.8267
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00327654.
391/391 [==============================] - 62s 159ms/step - loss: 0.3339 - acc: 0.8820 - val_loss: 0.5653 - val_acc: 0.8220
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0031138097.
391/391 [==============================] - 62s 158ms/step - loss: 0.3201 - acc: 0.8863 - val_loss: 0.5452 - val_acc: 0.8258
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0029664788.
391/391 [==============================] - 62s 158ms/step - loss: 0.3073 - acc: 0.8904 - val_loss: 0.5029 - val_acc: 0.8316
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.00283246.
391/391 [==============================] - 62s 159ms/step - loss: 0.2981 - acc: 0.8944 - val_loss: 0.5093 - val_acc: 0.8355
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0027100271.
391/391 [==============================] - 62s 158ms/step - loss: 0.2828 - acc: 0.9009 - val_loss: 0.5172 - val_acc: 0.8335
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.00259774.
391/391 [==============================] - 63s 160ms/step - loss: 0.2724 - acc: 0.9030 - val_loss: 0.5250 - val_acc: 0.8315
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0024943876.
391/391 [==============================] - 63s 160ms/step - loss: 0.2640 - acc: 0.9052 - val_loss: 0.5277 - val_acc: 0.8316
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0023989445.
391/391 [==============================] - 63s 161ms/step - loss: 0.2540 - acc: 0.9094 - val_loss: 0.5055 - val_acc: 0.8392
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.002310536.
391/391 [==============================] - 62s 159ms/step - loss: 0.2472 - acc: 0.9110 - val_loss: 0.5213 - val_acc: 0.8390
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0022284123.
391/391 [==============================] - 62s 159ms/step - loss: 0.2404 - acc: 0.9139 - val_loss: 0.5755 - val_acc: 0.8241
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.002151926.
391/391 [==============================] - 62s 159ms/step - loss: 0.2342 - acc: 0.9155 - val_loss: 0.5016 - val_acc: 0.8397
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.002080516.
391/391 [==============================] - 62s 159ms/step - loss: 0.2310 - acc: 0.9166 - val_loss: 0.5135 - val_acc: 0.8383
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0020136931.
391/391 [==============================] - 62s 159ms/step - loss: 0.2253 - acc: 0.9204 - val_loss: 0.5036 - val_acc: 0.8412
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0019510292.
391/391 [==============================] - 62s 159ms/step - loss: 0.2145 - acc: 0.9242 - val_loss: 0.5118 - val_acc: 0.8414
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0018921476.
391/391 [==============================] - 62s 159ms/step - loss: 0.2100 - acc: 0.9253 - val_loss: 0.5093 - val_acc: 0.8428
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.001836716.
391/391 [==============================] - 62s 158ms/step - loss: 0.2056 - acc: 0.9263 - val_loss: 0.5229 - val_acc: 0.8423
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0017844397.
391/391 [==============================] - 62s 159ms/step - loss: 0.2005 - acc: 0.9280 - val_loss: 0.5125 - val_acc: 0.8428
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0017350568.
391/391 [==============================] - 63s 160ms/step - loss: 0.1958 - acc: 0.9289 - val_loss: 0.5065 - val_acc: 0.8440
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0016883336.
391/391 [==============================] - 63s 160ms/step - loss: 0.1898 - acc: 0.9312 - val_loss: 0.5397 - val_acc: 0.8429
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0016440608.
391/391 [==============================] - 62s 159ms/step - loss: 0.1851 - acc: 0.9325 - val_loss: 0.5456 - val_acc: 0.8391
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0016020506.
391/391 [==============================] - 63s 160ms/step - loss: 0.1832 - acc: 0.9334 - val_loss: 0.5171 - val_acc: 0.8449
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0015621339.
391/391 [==============================] - 63s 161ms/step - loss: 0.1775 - acc: 0.9358 - val_loss: 0.5209 - val_acc: 0.8458
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0015241579.
391/391 [==============================] - 62s 159ms/step - loss: 0.1732 - acc: 0.9383 - val_loss: 0.5472 - val_acc: 0.8410
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0014879845.
391/391 [==============================] - 62s 159ms/step - loss: 0.1746 - acc: 0.9365 - val_loss: 0.5230 - val_acc: 0.8467
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0014534884.
391/391 [==============================] - 62s 159ms/step - loss: 0.1659 - acc: 0.9401 - val_loss: 0.5295 - val_acc: 0.8455
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0014205554.
391/391 [==============================] - 62s 159ms/step - loss: 0.1628 - acc: 0.9416 - val_loss: 0.5303 - val_acc: 0.8485
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0013890818.
391/391 [==============================] - 62s 159ms/step - loss: 0.1582 - acc: 0.9432 - val_loss: 0.5291 - val_acc: 0.8469
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0013589726.
391/391 [==============================] - 62s 159ms/step - loss: 0.1577 - acc: 0.9440 - val_loss: 0.5534 - val_acc: 0.8449
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.001330141.
391/391 [==============================] - 62s 159ms/step - loss: 0.1564 - acc: 0.9439 - val_loss: 0.5219 - val_acc: 0.8539
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0013025073.
391/391 [==============================] - 62s 159ms/step - loss: 0.1529 - acc: 0.9451 - val_loss: 0.5444 - val_acc: 0.8456
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0012759985.
391/391 [==============================] - 62s 159ms/step - loss: 0.1487 - acc: 0.9464 - val_loss: 0.5542 - val_acc: 0.8428
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0012505471.
391/391 [==============================] - 62s 159ms/step - loss: 0.1478 - acc: 0.9469 - val_loss: 0.5386 - val_acc: 0.8464
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0012260912.
391/391 [==============================] - 62s 158ms/step - loss: 0.1461 - acc: 0.9466 - val_loss: 0.5372 - val_acc: 0.8483
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0012025735.
391/391 [==============================] - 62s 159ms/step - loss: 0.1438 - acc: 0.9475 - val_loss: 0.5470 - val_acc: 0.8477
Model took 3128.50 seconds to train

Accuracy on test data is: 84.77
```
#### Accuracy of model on test data is 84.77 after 50 epochs and max validation accuracy of 85.39 was reached at the 46th epoch 

### Model without 1x1 bottlenecks (Output channel size and RF remain the same ) 

```
# Define the model
model = Sequential()

model.add(SeparableConv2D(64, 3, 3, border_mode='same',use_bias=False,depthwise_initializer=wt_init,pointwise_initializer=wt_init, input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

# output channel size = 32 x 32 , RF =3

model.add(SeparableConv2D(96, 3, 3,use_bias=False,depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

#output channel size = 30 x 30 , RF =5 


model.add(SeparableConv2D(96, 3, 3,use_bias=False,depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

# output channel size = 28 x 28, RF =7


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

# output channel size = 14 x 14 , RF = 8

model.add(SeparableConv2D(64, 3, 3, use_bias=False,border_mode='same',depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.03))

# output channel size = 14 x 14 , RF =12

model.add(SeparableConv2D(96, 3, 3,use_bias=False,border_mode='same',depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.03))

# output channel size = 14 x 14 , RF = 16

model.add(SeparableConv2D(192, 3, 3,use_bias=False,border_mode='same',depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

# output channel size = 14 x 14 , RF = 20


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

# output channel size = 7 x 7 , RF = 22


model.add(SeparableConv2D(96, 3, 3,use_bias=False, border_mode='same',depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.05))

# output channel size = 7 x 7 , RF =30 

model.add(SeparableConv2D(192, 3, 3,use_bias=False,depthwise_initializer=wt_init,pointwise_initializer=wt_init))
model.add(BatchNormalization())
model.add(Activation('relu'))

# output channel size = 5 x 5 , RF =38 

model.add(Convolution2D(10,1,1,use_bias=False,kernel_initializer=wt_init))

# output channel size = 5 x 5 , RF = 38

model.add(AveragePooling2D(5,5))

# output channel size = 1 x 1 , RF = 54 

model.add(Flatten())
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.02), metrics=['accuracy'])  

```

### Model training log for 50 epochs 

```
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.02.
391/391 [==============================] - 62s 160ms/step - loss: 1.6947 - acc: 0.3668 - val_loss: 4.0713 - val_acc: 0.3050
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0151630023.
391/391 [==============================] - 56s 144ms/step - loss: 1.2404 - acc: 0.5520 - val_loss: 1.5805 - val_acc: 0.5233
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0122100122.
391/391 [==============================] - 56s 143ms/step - loss: 1.0032 - acc: 0.6431 - val_loss: 1.2527 - val_acc: 0.5775
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0102197241.
391/391 [==============================] - 56s 143ms/step - loss: 0.8644 - acc: 0.6939 - val_loss: 0.9582 - val_acc: 0.6627
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0087873462.
391/391 [==============================] - 56s 143ms/step - loss: 0.7691 - acc: 0.7281 - val_loss: 1.0298 - val_acc: 0.6555
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0077071291.
391/391 [==============================] - 56s 143ms/step - loss: 0.6869 - acc: 0.7575 - val_loss: 0.7676 - val_acc: 0.7315
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.006863418.
391/391 [==============================] - 56s 143ms/step - loss: 0.6164 - acc: 0.7841 - val_loss: 0.8199 - val_acc: 0.7228
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0061862048.
391/391 [==============================] - 56s 143ms/step - loss: 0.5679 - acc: 0.8010 - val_loss: 0.7450 - val_acc: 0.7419
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0056306306.
391/391 [==============================] - 56s 144ms/step - loss: 0.5141 - acc: 0.8190 - val_loss: 0.7079 - val_acc: 0.7555
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0051666236.
391/391 [==============================] - 56s 143ms/step - loss: 0.4778 - acc: 0.8325 - val_loss: 0.6521 - val_acc: 0.7732
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0047732697.
391/391 [==============================] - 56s 143ms/step - loss: 0.4459 - acc: 0.8432 - val_loss: 0.6359 - val_acc: 0.7824
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0044355733.
391/391 [==============================] - 56s 143ms/step - loss: 0.4161 - acc: 0.8534 - val_loss: 0.6251 - val_acc: 0.7865
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0041425021.
391/391 [==============================] - 56s 143ms/step - loss: 0.3907 - acc: 0.8613 - val_loss: 0.6144 - val_acc: 0.7992
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0038857587.
391/391 [==============================] - 56s 143ms/step - loss: 0.3681 - acc: 0.8711 - val_loss: 0.5939 - val_acc: 0.8010
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0036589828.
391/391 [==============================] - 56s 142ms/step - loss: 0.3462 - acc: 0.8772 - val_loss: 0.5712 - val_acc: 0.8112
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0034572169.
391/391 [==============================] - 56s 143ms/step - loss: 0.3289 - acc: 0.8842 - val_loss: 0.5588 - val_acc: 0.8152
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00327654.
391/391 [==============================] - 56s 142ms/step - loss: 0.3084 - acc: 0.8908 - val_loss: 0.5887 - val_acc: 0.8116
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0031138097.
391/391 [==============================] - 56s 143ms/step - loss: 0.2951 - acc: 0.8940 - val_loss: 0.5755 - val_acc: 0.8226
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0029664788.
391/391 [==============================] - 56s 142ms/step - loss: 0.2796 - acc: 0.9002 - val_loss: 0.5369 - val_acc: 0.8282
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.00283246.
391/391 [==============================] - 56s 143ms/step - loss: 0.2737 - acc: 0.9013 - val_loss: 0.5447 - val_acc: 0.8274
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0027100271.
391/391 [==============================] - 56s 143ms/step - loss: 0.2575 - acc: 0.9082 - val_loss: 0.5837 - val_acc: 0.8166
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.00259774.
391/391 [==============================] - 56s 143ms/step - loss: 0.2479 - acc: 0.9118 - val_loss: 0.5538 - val_acc: 0.8223
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0024943876.
391/391 [==============================] - 56s 142ms/step - loss: 0.2365 - acc: 0.9144 - val_loss: 0.5519 - val_acc: 0.8264
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0023989445.
391/391 [==============================] - 56s 142ms/step - loss: 0.2254 - acc: 0.9197 - val_loss: 0.5570 - val_acc: 0.8288
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.002310536.
391/391 [==============================] - 56s 143ms/step - loss: 0.2186 - acc: 0.9220 - val_loss: 0.5710 - val_acc: 0.8278
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0022284123.
391/391 [==============================] - 56s 142ms/step - loss: 0.2109 - acc: 0.9239 - val_loss: 0.5305 - val_acc: 0.8368
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.002151926.
391/391 [==============================] - 56s 143ms/step - loss: 0.2019 - acc: 0.9278 - val_loss: 0.6023 - val_acc: 0.8217
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.002080516.
391/391 [==============================] - 55s 142ms/step - loss: 0.1928 - acc: 0.9311 - val_loss: 0.5992 - val_acc: 0.8241
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0020136931.
391/391 [==============================] - 56s 142ms/step - loss: 0.1901 - acc: 0.9316 - val_loss: 0.5544 - val_acc: 0.8379
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0019510292.
391/391 [==============================] - 56s 143ms/step - loss: 0.1797 - acc: 0.9355 - val_loss: 0.5551 - val_acc: 0.8386
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0018921476.
391/391 [==============================] - 57s 146ms/step - loss: 0.1780 - acc: 0.9362 - val_loss: 0.5602 - val_acc: 0.8359
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.001836716.
391/391 [==============================] - 57s 145ms/step - loss: 0.1708 - acc: 0.9381 - val_loss: 0.5796 - val_acc: 0.8318
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0017844397.
391/391 [==============================] - 57s 147ms/step - loss: 0.1647 - acc: 0.9402 - val_loss: 0.5845 - val_acc: 0.8292
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0017350568.
391/391 [==============================] - 57s 147ms/step - loss: 0.1622 - acc: 0.9418 - val_loss: 0.5580 - val_acc: 0.8403
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0016883336.
391/391 [==============================] - 57s 145ms/step - loss: 0.1573 - acc: 0.9424 - val_loss: 0.5918 - val_acc: 0.8317
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0016440608.
391/391 [==============================] - 57s 146ms/step - loss: 0.1509 - acc: 0.9463 - val_loss: 0.5905 - val_acc: 0.8359
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0016020506.
391/391 [==============================] - 57s 145ms/step - loss: 0.1491 - acc: 0.9473 - val_loss: 0.5867 - val_acc: 0.8377
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0015621339.
391/391 [==============================] - 57s 146ms/step - loss: 0.1438 - acc: 0.9491 - val_loss: 0.5929 - val_acc: 0.8353
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0015241579.
391/391 [==============================] - 57s 146ms/step - loss: 0.1408 - acc: 0.9491 - val_loss: 0.5967 - val_acc: 0.8347
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0014879845.
391/391 [==============================] - 57s 145ms/step - loss: 0.1380 - acc: 0.9503 - val_loss: 0.5885 - val_acc: 0.8410
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0014534884.
391/391 [==============================] - 56s 143ms/step - loss: 0.1338 - acc: 0.9523 - val_loss: 0.5937 - val_acc: 0.8401
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0014205554.
391/391 [==============================] - 56s 144ms/step - loss: 0.1302 - acc: 0.9517 - val_loss: 0.6355 - val_acc: 0.8292
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0013890818.
391/391 [==============================] - 56s 144ms/step - loss: 0.1254 - acc: 0.9539 - val_loss: 0.5907 - val_acc: 0.8421
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0013589726.
391/391 [==============================] - 56s 144ms/step - loss: 0.1271 - acc: 0.9540 - val_loss: 0.6048 - val_acc: 0.8381
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.001330141.
391/391 [==============================] - 56s 144ms/step - loss: 0.1238 - acc: 0.9552 - val_loss: 0.6072 - val_acc: 0.8386
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0013025073.
391/391 [==============================] - 56s 143ms/step - loss: 0.1195 - acc: 0.9560 - val_loss: 0.6235 - val_acc: 0.8357
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0012759985.
391/391 [==============================] - 56s 143ms/step - loss: 0.1217 - acc: 0.9563 - val_loss: 0.6209 - val_acc: 0.8390
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0012505471.
391/391 [==============================] - 56s 143ms/step - loss: 0.1178 - acc: 0.9575 - val_loss: 0.6142 - val_acc: 0.8401
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0012260912.
391/391 [==============================] - 56s 144ms/step - loss: 0.1131 - acc: 0.9592 - val_loss: 0.6375 - val_acc: 0.8305
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0012025735.
391/391 [==============================] - 57s 146ms/step - loss: 0.1116 - acc: 0.9602 - val_loss: 0.6382 - val_acc: 0.8394
Model took 2814.97 seconds to train

Accuracy on test data is: 83.94
```
#### The model without 1x1 bottleneck layers reached a max val accuracy of 84.21 in the 43rd epoch and val accuracy at the end of 50 epochs is 83.94 . These are also higher than the baseline model accuracy .

Additional Notes :
We can notice a reduction in time taken per epoch (about 9-10%) and also in overall time for 50 epochs between the two model versions. This is understandable given that we removed 2 1x1 bottlenecks layers.

We can also see that the gap between train and test accuracy for the model without bottlnecks is slightly larger than the model with bottlenecks. Perhaps the 1x1 bottlenecks also help in adding some level of regularization in addition to reducing the number of feature maps/parameters.

Link to Notebook : https://github.com/ravindrabharathi/Project1/blob/master/EIP4/session3/EIP4_Assignment3.ipynb
