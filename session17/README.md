## Training a CNN using TfRecords and tf.Data pipeline :

We have now added the ability to add random pad/crop , flip left to right and cutout augmenations and also plot misclassifed images

We can now do the following :

> #### Download Dataset and create tfrecords

>>tf_utils.get_cifar10_and_create_tfrecords()

> #### Create tf dastasets for training and testing 

>>train_ds=tf_utils.get_train_ds()

>>test_ds=tf_utils.get_eval_ds()

> #### plot augmented images from batched dataset


>>tf_utils.plot_cifar10_files(train_ds)

> #### plot misclassified images 

>>wrong_indices, wrong_labels, true_labels, wrong_set = tf_utils.get_misclassified_images(model,test_ds)

>>tf_utils.plot_misclassified_images(wrong_indices, wrong_labels, true_labels, wrong_set,52)

Profiling:
To be added 
