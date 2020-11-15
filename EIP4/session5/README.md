# Build a model to classify a Person's attributes

We will build a model that classifies the attributes of a person seen in a photo. As shown in the image below, in addition to the person's attributes , we also try to categorize the image quality .

![classify](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/classify.png) 

### Link to Notebook : https://github.com/ravindrabharathi/Project1/blob/master/EIP4/session5/010_PersonAttributes.ipynb

## Image Data Preparation

### Crop images to remove black padding 
The dataset contained images are of size 224x224 and padded with black pixels . 
We will remove the black pixels and use the resulting images for our training . 
Visual inspection of the images shows that many images are 120 pixels or less in actual width and rest is black padding. So take this route of cropping out the black pixels . We could then resize the actual image to various proportions using a datagenerator 
The code used for cropping images is at 
https://github.com/ravindrabharathi/Project1/blob/master/EIP4/session5/Crop_images_PersonAttributes.ipynb

![crop](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/image_cropped.png) 

## Data Labels 
Original labels in the dataset are strings as shown below 

![originasl_df](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/original_df.png) 

We used a integer based categories for these labels and transformed them as shown below. 
This lends itself well to the use of sparse_cross_entropy loss function that we used for training. 
![transformed_df](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/transformed_df.png) 


## Data split for train and test sets 
scikit leran train_test_split was used for setting aside 15% data for validation . 
Random seed of 2019 was used such that the split happens exactly the same way every time the split function is called . 
This ensures that there is no mixing of train and test data

```
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.15,shuffle=True,random_state=2019)
train_df.shape, val_df.shape

```
### Imbalance in Dataset:
The dataset was highly imbalanced with some classes having the majority of samples .

This can be seen below in the figure in the train and validation data distribution. 
We tried to address this by repeating some of the samples from the train data more than the others .
We made sure that this oversampling used only data from training set 

![data_imbalance](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/data_imbalance.png) 

## Model 

We built a model that had 3 branches . Main Branch for everything except age and footwear. 
Age Branch that used the top part of our image for age classification
Footwear branch that used the bottom part of the image for footwear classification.

The output from the three branches were mixed at the end for each of the heads used for classification 

A simplified view of the model is as below 


![model](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/model_simplified_view.png)

### model code

Main Branch

```
def build_main_branch(inp,f):
  layer1=conv(inp,f,k1=3,k2=3,s1=1,s2=1,padng='same')
  layer1=conv(layer1,f,k1=3,k2=3,s1=1,s2=1,padng='same')
 
  layer2=conv(layer1,f,k1=3,k2=3,s1=1,s2=1,padng='same',dilation=True)
 
  layer1=Concatenate()([layer1,layer2])
 
  layer1=conv(layer1,f,k1=1,k2=1,s1=1,s2=1,padng='same')

  res1=resBlk(layer1,f*2,3,True,True,True)
  res2=resBlk(res1,f*4,3,False,True,True)
  res3=resBlk(res2,f*4,3)
  res4=resBlk(res3,f*8,3)
 
  last_block=conv(res4,f*16,k1=3,k2=3,s1=2,s2=1,padng='same')
  last_block=Conv2D(f*8,1,use_bias=False,kernel_initializer=wt_init, kernel_regularizer=reg)(last_block)
  last_block= BatchNormalization(momentum=0.9, epsilon=1e-5)(last_block)
  last_block= LeakyReLU(alpha=0.2,name='main_last_lr')(last_block)
  layer2=GlobalAveragePooling2D()(last_block)
 
  return layer2

```
Age Branch:
```
def build_age_branch(inp,f) :
  
  layer1=conv(inp,f,k1=3,k2=3,s1=1,s2=1,padng='same')
  layer1=conv(layer1,f,k1=3,k2=3,s1=1,s2=1,padng='same')
  
  layer2=conv(layer1,f,k1=3,k2=3,s1=1,s2=1,padng='same',dilation=True)
  
  layer1=Concatenate()([layer1,layer2])
  
  layer1=conv(layer1,f,k1=1,k2=1,s1=1,s2=1,padng='same')

  res1=resBlk(layer1,f*2,3,True,True,True)
  res2=resBlk(res1,f*4,3,False,True,True)
  res3=resBlk(res2,f*4,3)
  res4=resBlk(res3,f*8,3)

  last_block=conv(res4,f*16,k1=3,k2=3,s1=1,s2=1,padng='same')
  last_block=Conv2D(f*8,1,use_bias=False,kernel_initializer=wt_init, kernel_regularizer=reg)(last_block)
  last_block= BatchNormalization(momentum=0.9, epsilon=1e-5)(last_block)
  last_block= LeakyReLU(alpha=0.2,name='age_last_lr')(last_block)
  layer2=GlobalAveragePooling2D()(last_block)
  return layer2
  ```
  Foot Branch is similar to Age Branch
  
  Head code 
  ```
  def build_head(name, in_layer,add_dense=True):

  if add_dense:
    last_layer=  Dense(64, kernel_initializer=wt_init, use_bias=False,kernel_regularizer=reg)(in_layer)
    last_layer= BatchNormalization(momentum=0.9, epsilon=1e-5)(last_layer)
    last_layer= LeakyReLU(alpha=0.2,name=name+'_head_last_lr')(last_layer)
    
  else:
    last_layer=in_layer 

  last_layer =Dense(num_units[name], use_bias=False,kernel_initializer=wt_init)(last_layer)   
  
  out=Activation('softmax',name=f"{name}_output") (last_layer)
  return out 
  ```
  
  Main model build code 
  ```
  def build_model():
  f=16
  global c_c
  c_c=0 # used for naming the conv layers
  inp=Input(shape=(None,None,3))
  
  inp2=Lambda(lambda x:crop_top1(x))(inp)
  inp3=Lambda(lambda x:crop_botm1(x))(inp)
  main_branch=build_main_branch(inp,f)
  age_branch=build_age_branch(inp2,f)
  foot_branch=build_foot_branch(inp3,f)


  main_branch1=Add()([main_branch,age_branch,foot_branch])

  main_branch2=Add()([main_branch,age_branch])

  main_branch3=Add()([main_branch,foot_branch])
  

  gender = build_head("gender", main_branch1)
  image_quality = build_head("imagequality", main_branch1)
  
  age = build_head("age", main_branch2)
  weight = build_head("weight", main_branch1)
  bag = build_head("carryingbag", main_branch1)
  
  footwear = build_head("footwear", main_branch3)
  
  emotion = build_head("emotion", main_branch2)
  pose = build_head("bodypose", main_branch)
  model=Model(inputs=[inp],outputs=[gender,image_quality,age,weight,bag,footwear,emotion,pose])
  
  model.summary()
  return model 
  
  ```
  
  ## Datagenerator : 
  ImageDatagenerator was used for generating training and validation samples 
  
  ```
  train_datagen = ImageDataGenerator(
        preprocessing_function=norm_mean
        )

val_datagen = ImageDataGenerator(
    preprocessing_function=norm_mean
    )

train_gen1 = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='./',
        x_col="image_path",
        y_col=['gender',	'imagequality',	'age'	,'weight'	,'carryingbag',	'footwear',	'emotion'	,'bodypose'],
        target_size=(224, 112), interpolation='bicubic',
        batch_size=32,
        class_mode='multi_output')

val_gen1 = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory='./',
        x_col="image_path",
        y_col=['gender',	'imagequality',	'age'	,'weight'	,'carryingbag',	'footwear',	'emotion'	,'bodypose'],
        target_size=(224, 112),interpolation='bicubic',
        batch_size=64,
        class_mode='multi_output')
  ```
  
  We varied the target sizes to 1/2 , 3/4 dimensions for some epochs 
  
  ### Model Compilation / Optimizer/ Loss function
  We used SGD optimizer with 0.9 momentum , nesterov True and varied the lr between 0.001 and 0.0001
  
  Loss function used was sparse_categorical_crossentropy 
  
  We tried using loss weights for penalising age , imagequality losses more but didn't see immediate results 
  
  ### Training and results :
  
  We trained the model for about 200 epochs and saved the model by monitoring gender-accuracy. 
  Ideally we should have saved the model at end of each epoch to maybe look for the best combination but we chose gender accuracy due to space constraints on Google drive 
  
  Model evaluation result at the end of 200 epochs using a odel saved at 175th epoch
  
  ```
  {'age_output_acc': 38.3595,
 'bodypose_output_acc': 85.7564,
 'carryingbag_output_acc': 63.556,
 'emotion_output_acc': 66.2083,
 'footwear_output_acc': 62.6228,
 'gender_output_acc': 89.4892,
 'imagequality_output_acc': 54.7642,
 'weight_output_acc': 57.6621}
 ```
  
  Epoch 112 results seemed to be a good combination but unfortunately we did not save this model run as we were tracking only gender accuracy
```
val_gender_output_acc: 0.8399 - 

val_imagequality_output_acc: 0.5560 - 

val_age_output_acc: 0.4332 - 

val_weight_output_acc: 0.6100 - 

val_carryingbag_output_acc: 0.6483 - 

val_footwear_output_acc: 0.6356 - 

val_emotion_output_acc: 0.7019 - 

val_bodypose_output_acc: 0.8433
```

### Some more experiments that could be done :
Freeze main branch or any of the other two branches and train to see if there is improvement in age accuracy or imagequality accuracy . Unfortunately we could not complete these experiments in the current trial as Colab GPU was denied due to long running training tasks 
  
### Link to Notebook : 
https://github.com/ravindrabharathi/Project1/blob/master/EIP4/session5/010_PersonAttributes.ipynb  
  
 

