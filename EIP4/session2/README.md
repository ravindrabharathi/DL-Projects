# Train a CNN with less than 15k parameters to get 99.40 accuracy 

### link to notebook : 

>>>> https://github.com/ravindrabharathi/Project1/blob/master/EIP4/session2/Ninth.ipynb

## Logs from training 20 epochs 

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 15s 257us/step - loss: 0.1724 - acc: 0.9455 - val_loss: 0.0466 - val_acc: 0.9842
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0589 - acc: 0.9819 - val_loss: 0.0316 - val_acc: 0.9893
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0452 - acc: 0.9857 - val_loss: 0.0298 - val_acc: 0.9913
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 9s 150us/step - loss: 0.0377 - acc: 0.9877 - val_loss: 0.0294 - val_acc: 0.9909
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 9s 150us/step - loss: 0.0337 - acc: 0.9896 - val_loss: 0.0253 - val_acc: 0.9921
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 9s 155us/step - loss: 0.0294 - acc: 0.9905 - val_loss: 0.0231 - val_acc: 0.9926
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0271 - acc: 0.9912 - val_loss: 0.0203 - val_acc: 0.9938
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0251 - acc: 0.9916 - val_loss: 0.0276 - val_acc: 0.9916
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 9s 152us/step - loss: 0.0233 - acc: 0.9924 - val_loss: 0.0209 - val_acc: 0.9936
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 9s 150us/step - loss: 0.0227 - acc: 0.9926 - val_loss: 0.0236 - val_acc: 0.9933
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0209 - acc: 0.9930 - val_loss: 0.0247 - val_acc: 0.9931
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 9s 152us/step - loss: 0.0202 - acc: 0.9932 - val_loss: 0.0206 - val_acc: 0.9939
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0181 - acc: 0.9940 - val_loss: 0.0192 - val_acc: 0.9942
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0176 - acc: 0.9943 - val_loss: 0.0210 - val_acc: 0.9936
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0181 - acc: 0.9942 - val_loss: 0.0218 - val_acc: 0.9940
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 9s 150us/step - loss: 0.0163 - acc: 0.9948 - val_loss: 0.0200 - val_acc: 0.9947
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0147 - acc: 0.9952 - val_loss: 0.0196 - val_acc: 0.9945
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0154 - acc: 0.9950 - val_loss: 0.0190 - val_acc: 0.9953
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0151 - acc: 0.9948 - val_loss: 0.0210 - val_acc: 0.9937
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 9s 151us/step - loss: 0.0139 - acc: 0.9952 - val_loss: 0.0217 - val_acc: 0.9940
<keras.callbacks.History at 0x7fbc72124c18>


## Model evaluate score 

[0.021709251910779496, 0.994]

## Strategy
1. Reduce number of kernels in first conv layer to 8 
2. Insert another MaxPooling2D layer 3 layers above the last layer
3. Change last layer kernel size to 5,5 instead of 4,4  
4. Make sure there is no BatchNormalization , Dropout, activation in last layer 
5. Make sure that there are no Dense layers used 
6. Remove Bias param by setting use_bias to false 


