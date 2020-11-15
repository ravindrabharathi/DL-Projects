# Use a pretrained Resnet model to classify CIFAR100 images

We will use a ResNet34 pretrained model from https://github.com/qubvel/classification_models

We will use Resnet34 model to try and achieve 80% validation accuracy . Since pretrained weights are only available for imagenet and models expect a 224x224 image size , we will resize the cifar100 images to 224x224 while training .

In the pretrained model we will remove the top prediction layers and freeze the last 11 layers . 
We will add a GlobalAveragepooling2D layer , a dense layer and a softmax activation to form our prediction layer for cifar100.
The first part will be to train with the frozen layers in base model . 
After training for about 30 epochs , we will unfreeze the rest of the layers and train further .

### Model definition

```
# build model
from keras.layers import GlobalAveragePooling2D, Add, Lambda, Dense, GlobalMaxPooling2D

#base modek from REsnet34 
base_model = ResNet34(input_shape=(224,224,3), weights='imagenet', include_top=False)

#Freeze all but last 11 layers 
for layer in base_model.layers[:-11]:
  layer.trainable=False
for layer in base_model.layers:
    print(layer, layer.trainable) 

#Add our own Top/Prediction layers 
x = GlobalAveragePooling2D()(base_model.output)



x= Dense(num_classes,use_bias=False)(x)

output = keras.layers.Activation('softmax')(x)

model = keras.models.Model(inputs=[base_model.input], outputs=[output])

```

### Training

Model was trained for 30 epochs with layers frozen .

Accuracy at Epoch 30:
```
Epoch 30/30
391/391 [==============================] - 121s 309ms/step - loss: 1.3943 - acc: 0.6102 - val_loss: 2.9288 - val_acc: 0.4246

Epoch 00030: val_acc did not improve from 0.43420
```

Then layers were unfrozen and trained for a further 100 epochs 
```
for layer in model.layers:
  layer.trainable=True
```

Model accuracies at various epochs :

Overall Epoch 65:
```
Epoch 35/100
391/391 [==============================] - 177s 454ms/step - loss: 0.0800 - acc: 0.9777 - val_loss: 0.8207 - val_acc: 0.8023

Epoch 00035: val_acc improved from 0.79800 to 0.80230, saving model to /gdrive/My Drive/EVA/session20/best_model2.h5
```
Overall Epoch 130:
```
Epoch 00100: val_acc did not improve from 0.81310
```
Trained another 100 epochs but runtime disconnected after epoch 27 (Overall Epoch 157) :
```
Epoch 27/100
391/391 [==============================] - 565s 1s/step - loss: 0.0015 - acc: 0.9997 - val_loss: 0.7848 - val_acc: 0.8152

Epoch 00027: val_acc improved from 0.81490 to 0.81520, saving model to /gdrive/My Drive/EVA/session20/best_model2.h5
```

Max Vall accuracy after 157 epochs : 81.52 


### Model score 
```
score=model.evaluate_generator(validation_generator)

print('validation loss =',score[0] , ', Validation accuracy =',score[1])
```
##### validation loss = 0.7847665718078614 , Validation accuracy = 0.8152


##### Link to Notebook : https://github.com/ravindrabharathi/Project1/blob/master/session20/003_cifar100.ipynb
