
import tarfile
import os
import sys
from urllib.request import urlretrieve
import numpy as np
import requests

from tqdm import tqdm_notebook as tqdm

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange

import numpy as np

import time
import functools
import psutil


## for cifar10 
tar = 'cifar-10-python.tar.gz'
url = 'https://www.cs.toronto.edu/~kriz/' + tar

num_classes=10
batch_size=128
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

#import tf if not defined 
try:
  tf
except NameError:
    import tensorflow as tf
else:
    pass


#for tf2 eager execution is enabled by default 
# for lower versions enable eager execution
if (int(str(tf.__version__)[:1])<2):
  tf.compat.v1.enable_eager_execution()

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer
  
@timer
def get_cpu_num():
  
  cpu_num=psutil.cpu_count()
  
  return cpu_num
  


#get the cpu cores on current env 
CPU_CORES=get_cpu_num() #get_cpu_cores()  

# features of a single record
rec_features = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

###
#TRANSFORM FUNCTIONS
###

def center_pad_crop(image,padding=4):
  image=tf.pad(image,[(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='reflect')
  central_fraction=((tf.shape(image)[1]-2*padding)/tf.shape(image)[1])
  print(central_fraction)
  image=tf.image.central_crop(image,central_fraction)

  return image

def random_pad_crop(image,padding=4):
  shp=tf.shape(image)
  
  image=tf.pad(image,[(0, 0), (padding, padding), (padding, padding), (0, 0)], mode='reflect')
  
  image=tf.image.random_crop(image,size=shp)
  return image

def flip_left_right(image):
  return tf.image.random_flip_left_right(image)

def cutout(img, prob=100, size=8, min_size=5, use_fixed_size=True):
  return tf.cond(tf.random.uniform([], 0, 100) > prob, lambda: img , lambda: get_cutout(img,prob,size,min_size,use_fixed_size))



def get_cutout(img,prob=50,size=8,min_size=5,use_fixed_size=True):
  
  shp=tf.shape(img)
  
  
  height = width = tf.shape(img)[1]
  
  channel = 3
  
  

  #get cutout size and offsets 
  if (use_fixed_size==True):
    s=size
  else:  
    s=tf.random.uniform([], min_size, size, tf.int32) # use a cutout size between 5 and size 

  x1 = tf.random.uniform([], 0, height+1-s , tf.int32) # get the x offset from top left
  y1 = tf.random.uniform([], 0, width+1-s , tf.int32) # get the y offset from top left 
  
  # create the cutout slice and the mask 
  img1 = tf.ones_like(img)  
  print(tf.shape(img1))
  cut_slice = tf.slice(
  img1,
  [0,x1, y1, 0],
  [shp[0],s, s, 3]
     )
  
  
  #create mask similar in shape to input image with cutout area having ones and rest of the area padded with zeros 
  mask = tf.image.pad_to_bounding_box(
    cut_slice,
    x1,
    y1,
    height,
    width
  )
  
  
  #invert the zeros and ones 
  mask = tf.ones_like(mask ) - mask
  
  print(tf.shape(mask))
  
  tmp_img = tf.multiply(img,mask)
  
  cut_img =tmp_img
   
  return cut_img

def aug1(image):
  return cutout(flip_left_right(random_pad_crop(image)))

def aug2(image):
  return cutout(flip_left_right(central_pad_crop(image)))
###
#FUNCTIONS TO DOWLOAD DATASET AND CREATE TFRECORDS
###

#download a file and write to disk  
@timer
def download_file(url, dst):
    file_size = int(requests.head(url).headers["Content-Length"])

    pbar = tqdm(
        total=file_size, initial=0,
        unit='B', unit_scale=True, desc=url.split('/')[-1])

    req = requests.get(url, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=10 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(10 * 1024)
    pbar.close()
    return file_size

#dowload cifar10 data
@timer
def download_cifar10_files():
    path = './'
    if tar not in os.listdir(path):
        download_file(url, tar)
    else:
        print('dataset archive file exists!')

#extract cifar 10 data files downloaded archive 
@timer
def extract_cifar10_files():
    data = './cifar10_data/'
    if os.path.exists(data + 'cifar-10-batches-py/test_batch'):
        print(data + 'cifar-10-batches-py/', 'is not empty and has test_batch file!')

    else:
        tarfile.open(tar, 'r:gz').extractall(data)
        print('Done')

#get filenames in cifar 10 data and split them as train and eval set 
@timer
def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 6)]
    file_names['eval'] = ['test_batch']
    return file_names

#read the data files 
@timer
def read_pickle_from_file(filename):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding='bytes')
        else:
            data_dict = pickle.load(f)
    return data_dict


#tfrecord features for label and image in cifar 10 dataset 
#label
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#image
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#convert downloaded data files to tfrecords 
@timer
def convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            # print(input_file)
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b'data']
            labels = data_dict[b'labels']
            num_entries_in_batch = len(labels)
            # print(num_entries_in_batch)

            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(data[i].tobytes()),
                        'label': _int64_feature(labels[i])
                    }))
                record_writer.write(example.SerializeToString())

#function to check if tfrecords exist , else create them 
@timer
def create_tf_records(data_dir='./cifar10_data', output_dir='./', overwrite=False):
    file_names = _get_file_names()

    input_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = mode + '.tfrecords'
        if ((output_file in os.listdir(output_dir)) & (overwrite == False)):
            print(output_file, 'exists!')
        else:
            output_file = os.path.join(output_dir, output_file)
            try:
                os.remove(output_file)
            except OSError:
                pass
            # Convert to tf.train.Example and write the to TFRecords.
            convert_to_tfrecord(input_files, output_file)
            print('Done!')

# putting it all together -- download data and create tfrecords 
@timer
def get_cifar10_and_create_tfrecords():
    download_cifar10_files()
    extract_cifar10_files()
    create_tf_records()

###
#FUNCTIONS TO READ TFRECORDS AND CREATE DATASETS  
###   

# function to parse a single record in tfrecords
def _parse_record_function(im_example):
    return tf.io.parse_single_example(im_example, rec_features)

#parse a batch of records if you batch before map
def parse_batch(batch_of_records):
    records=tf.io.parse_example(batch_of_records,rec_features)
    image = tf.io.decode_raw(records['image'], tf.uint8)
    
    #image.set_shape([batch_size * 32 * 32 * 3]) # refer to https://stackoverflow.com/questions/35451948/clarification-on-tf-tensor-set-shape
    image=tf.transpose(tf.reshape(image,[batch_size,3,32,32]),[0,2,3,1])
    #cast image as float32 as the model requires it
    image = tf.cast(image,tf.float32)
    #augment image if needed
    #send image to augment fn here 
    #image=aug_fn(image)
    ##

    label = tf.cast(records['label'], tf.int32)
    label = tf.one_hot(label, num_classes)

    return image, label
  
#parse a batch of records if you batch before map
def parse_batch_distort(batch_of_records):
    records=tf.io.parse_example(batch_of_records,rec_features)
    image = tf.io.decode_raw(records['image'], tf.uint8)
    
    #image.set_shape([batch_size * 32 * 32 * 3]) # refer to https://stackoverflow.com/questions/35451948/clarification-on-tf-tensor-set-shape
    image=tf.transpose(tf.reshape(image,[batch_size,3,32,32]),[0,2,3,1])
    #cast image as float32 as the model requires it
    image = tf.cast(image,tf.float32)
    
    #image augmenation 
    print('distorting image')
    image=aug1(image)
    

    label = tf.cast(records['label'], tf.int32)
    label = tf.one_hot(label, num_classes)

    return image, label  

    


#function to parse a single record and prepare an image/label set for training / evaluation
def parse_record(im_example):
    record = tf.io.parse_single_example(im_example, rec_features)
    
    image = tf.io.decode_raw(record['image'], tf.uint8)
    image.set_shape([32 * 32 * 3])
    image = tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0])
    #image = tf.reshape(image,[32,32,3]) #check this ..this doesn't seem to give the right image 
    image = tf.cast(image,tf.float32)
    #augment image if needed
    #send image to augment fn here 
    #image=aug_fn(image)
    ##

    label = tf.cast(record['label'], tf.int32)
    label = tf.one_hot(label, num_classes)

    return image, label

#function to create train and eval datasets 
@timer
def create_train_eval_datasets():
    train_records = tf.data.TFRecordDataset('./train.tfrecords')
    eval_records = tf.data.TFRecordDataset('./eval.tfrecords')
    train_dataset = train_records.map(_parse_record_function)
    eval_dataset = eval_records.map(_parse_record_function)

    return train_dataset, eval_dataset

#function to create dataset from tfrecords
@timer
def get_tf_dataset(recordsfile, batch_size, shuffle=False,distort=False):
  #create dataset from tfrecords file  
  files=tf.data.Dataset.list_files(recordsfile)
  dataset = files.interleave(tf.data.TFRecordDataset,cycle_length=4)
  #shuffle 
  if shuffle:
    dataset = dataset.shuffle(buffer_size=10*batch_size)
  #repeat
  dataset = dataset.repeat()
  #batch before map is recommended for speed
  #refer https://stackoverflow.com/questions/50781373/using-feed-dict-is-more-than-5x-faster-than-using-dataset-api
  # batch the records
  dataset = dataset.batch(batch_size=batch_size)
  #parse the records - map to parse function
  #setting num_parallel_calls to a value much greater than the number of available CPUs 
  #can lead to inefficient scheduling, resulting in a slowdown
  if distort:
    print('distorting...')
    dataset = dataset.map(map_func=parse_batch_distort,num_parallel_calls=CPU_CORES)
  else:
    dataset = dataset.map(map_func=parse_batch,num_parallel_calls=CPU_CORES)
  #prefetch elements from the input dataset ahead of the time they are requested
  dataset = dataset.prefetch(buffer_size=1)
  
  return dataset

#create dataset and return an iterator for dataset 
@timer
def get_tf_dataset_in_batches(recordstype='train', batch_size=128, shuffle=False,distort=False):
  #switch file name based on train or test data
  if recordstype == 'train':
    recordsfile = './train.tfrecords'
  else:
    recordsfile = './eval.tfrecords'
  #create dataset 
  dataset = get_tf_dataset(recordsfile, batch_size,shuffle,distort)
  # tf version is 2 return dataset , else return an iterator 
  if (int(str(tf.__version__)[:1])<2):
    #create an iterator for the dataset   
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return iterator
  else:
    return dataset

#create train data 
@timer
def get_train_ds(batch_size=128):
    train_ds = get_tf_dataset_in_batches('train', batch_size, True,True)
    return train_ds

#create test data
@timer
def get_eval_ds(batch_size=128):
    test_ds = get_tf_dataset_in_batches('test', batch_size)
    return test_ds

#function to plot n images from dataset
def plot_cifar10_files(dataset,n=5):
  import matplotlib.pyplot as plt
  records=dataset.take(1)
  #print(records)

  for record in records:
    image_batch,label_batch=record
    
    image_batch=image_batch.numpy()
    for i in range(n):
      
      plt.imshow(image_batch[i].astype('uint8'))
      plt.show()
      
#function to get misclssified images
def get_misclassified_images(model,test_ds):
  num_steps=np.ceil(10000/batch_size)
  pred=model.predict(test_ds,steps =num_steps, verbose=1)
  pred2=np.argmax(pred,axis=1)
  wrong_set=[]
  correct_set=[]
  wrong_labels=[]
  true_labels=[]
  wrong_indices=[]
  
  c=0
  y=None
  
  for record in test_ds.take(num_steps):
    if c==0:
      x=record[0].numpy()
      y=record[1].numpy()
      c+=1
    else:
      y= np.vstack((y,record[1].numpy()))
      x= np.vstack((x,record[0].numpy()))
  y=y[:10000]
  pred2=pred2[:10000]
  x=x[:10000]
  for i in range(10000):
    y1=np.argmax(y[i])
    if pred2[i]==y1:
      correct_set.append(x[i])
    else:
      wrong_indices.append(i)
      wrong_labels.append(class_names[pred2[i]])
      true_labels.append(class_names[y1])
      wrong_set.append(x[i])
  
  return wrong_indices, wrong_labels, true_labels, wrong_set  


#function to display images 
import matplotlib.pyplot as plt
def displayRow(images,titles):
  n=len(images)
  m=4-n
  
  if n<4:
    for j in range(m):
      
      dummy_image=(np.ones([32,32,3]))
      dummy_image=dummy_image*255
      images.append(dummy_image)
      titles.append('')
      
      
      
  
  
  fig = plt.figure(1, (13,13))
  
  grid = ImageGrid(fig, 111,  
                 nrows_ncols=(1,len(images)),  
                 axes_pad=1,label_mode="1"  
                 )
  
  for i in range(len(images)):
    grid[i].imshow(images[i].astype('uint8'))
    grid[i].set_title(titles[i])
    grid[i].axis('off')
  plt.show()
  
  
  display(HTML("<hr size='5' color='black' width='100%' align='center' />"))

#function to grop misclassified records in rows
from mpl_toolkits.axes_grid1 import ImageGrid

from IPython.core.display import display, HTML

def plot_misclassified_images(wrong_indices,wrong_labels,true_labels,wrong_set,num_images=50):
  
  heder="<h2 align='center'>First "+str(num_images)+" misclassified images</h2><hr size='5' color='black' width='100%' align='center' />"
  display(HTML(heder))
  
  for i in range(0,num_images,4):
    images=[]
    titles=[]
    
    for j in range(4):
      
      if (i+j)<num_images:
        
        images.append(wrong_set[i+j])
        title=str(wrong_indices[i+j])+':'+true_labels[i+j]+'\n predicted as \n'+wrong_labels[i+j]
    
        titles.append(title)
    
    
    
    
    displayRow(images,titles)
  
  
