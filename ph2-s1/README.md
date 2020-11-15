# Train a Neural Network to classify IMDB reviews as postive/negative 

### Reference code : 
We use the reference code from François Chollet's Deep Learning with Python book to train a Neural Network to classify IMBD reviews 

### IMDB Dataset:
[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) is a dataset for binary sentiment classification containing a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 

It was first used in the paper 'Learning Word Vectors for Sentiment Analysis' by Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011) for The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

### GloVe Embeddings

There are various precomputed databases of word embeddings that you can download and use in a Keras Embedding layer. 
One of them is Global Vectors for Word Representation (GloVe, https://nlp.stanford.edu/projects/glove), 
which was developed by Stanford researchers in 2014. This embedding technique is based on factorizing a matrix of word 
co-occurrence statistics. Its developers have made available precomputed embeddings for millions of English tokens, 
obtained from Wikipedia data and Common Crawl data.

### Model :
Our model is a sequentila model consisting of an embedding layer , a Dense layer with reLu activation and 
a Classification layer with Sigmoid activation

```python

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

```

### Number of training samples : 8000
```
maxlen = 100  # We will cut reviews after 100 words
training_samples = 8000  # We will be training on 8000 samples
validation_samples = 10000  # We will be validating on 10000 samples
max_words = 10000  # We will only consider the top 10,000 words in the dataset
```


### Traning model with pretrained GloVe embeddings :

We trained the model with 8000 training samples and pretrained GloVe embeddings.

In this case we pretrained word embeddings into the Embedding layer and freeze the embedding layer 

```
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

```

We get a validation accuracy of about 70%. 

The accuracy and loss plots are as shown below 

![accuracy-plot1](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/acc-plot1.png)

![loss-plot1](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/loss-plot1.png)

### Evaluation score on Test Data:
```
model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)

25000/25000 [==============================] - 1s 40us/step
[0.8920598086166381, 0.68976]
```

### Training without pretrained GloVe Embeddings : 

In this case the Model learns taskspecific embedding of input tokens and validation accuracy was around 81%

Accuracy and Loss plots are as shown below

![accuracy-plot2](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/acc-plot2.png)

![loss-plot2](https://raw.githubusercontent.com/ravindrabharathi/eip3/master/images/loss-plot2.png)

### Evaluation score on test data:

```
model.evaluate(x_test, y_test)
25000/25000 [==============================] - 1s 48us/step
[0.9959731257635355, 0.81008]
```

### summary :

We trained a model to classify IMBD reviews 

1. using pretrained GloVe word embeddings with 8000 training samples and reached a test accuracy of about 69% 

2. Without pretrained GloVe word embeddings using 8000 training samples and reached a test accuracy of about 81% 

In the original example from the book using 200 samples, the accuracy reached was around 50-55% and pretrained word embeddings seemed to perform slightly better. 
But with a larger data size of 8000 samples we find that the model performed better when it learns task-specific embedding of input tokens without using the pre-trained embeddings .
