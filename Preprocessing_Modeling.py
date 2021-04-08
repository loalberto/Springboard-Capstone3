#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# In[1]:


import os
from tensorflow.keras.preprocessing import image
import numpy as np
import multiprocessing 
import random
import pandas as pd
import multiprocessing
import gc
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[2]:


def get_file_names(s):
    # retrieves all the filenames in a list of strings
    path = './transformed_images/{}'.format(s)
    vals = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if os.path.getsize(path + '/'+ filename) == 0 or filename == '.DS_Store':
                continue
            vals.append(filename)
    return sorted(vals)


# In[3]:


def tonp(func, list_of_images, size=(300, 300)):
    # for img in list_of_images:
    path = func(list_of_images)
    # Transforming all the images to size 400x400
    current_img = image.load_img(path, target_size=size, color_mode='grayscale')
    # makes a matrix
    img_ts = image.img_to_array(current_img)
    # converts to a vector
    img_ts = [img_ts.ravel()]
    current_img.close()
    try:
        # Brings all the new vectors into one giant array
        full_mat = np.concatenate((full_mat, img_ts))
    except UnboundLocalError:
        full_mat = img_ts
    return full_mat


# In[4]:


def tonp_wrapper(args):
    return tonp(*args)


# In[5]:


def get_cat_filepath(img_name):
    # Returns the filepath of a given string
    return './transformed_images/Cat/{}'.format(img_name)


# In[6]:


def get_dog_train_filepath(img_name):
    # Returns the filepath of a given string
    return './transformed_images/DogTrain/{}'.format(img_name)


# In[7]:


def get_dog_test_filepath(img_name):
    # Returns the filepath of a given string
    return './transformed_images/DogTest/{}'.format(img_name)


# In[8]:


def display_image_np(np_array):
    # The functiton takes in an np_array to display the image
    # This will display the image in grayscale
    plt.imshow(np_array, vmin=0, vmax=255, cmap='Greys_r')
    plt.axis('off')
    plt.grid(True)
    plt.show()
    plt.show()


# In[9]:


def set_up_data(cat_filenames, dogtrain_filenames, dogtest_filenames, sample_amount=5000):
    cat_data = []
    dogtrain_data = []
    dogtest_data = []
    # for i in range(len(cat_filenames)):
    for i in range(sample_amount):
        cat_data.append(tonp(get_cat_filepath, cat_filenames[i]))
    # for i in range(len(dogtrain_filenames)):
    for i in range(sample_amount):
        dogtrain_data.append(tonp(get_dog_train_filepath, dogtrain_filenames[i]))
    # for i in range(len(dogtest_filenames)):
    for i in range(sample_amount):
        dogtest_data.append(tonp(get_dog_test_filepath, dogtest_filenames[i]))
    dog_data = np.concatenate((dogtest_data, dogtrain_data))
    del dogtest_data
    del dogtrain_data
    gc.collect()
    sample_cat = random.sample(cat_data, sample_amount)
    cat_label = np.array([1 for _ in range(len(cat_data))])
    dog_label = np.array([0 for _ in range(len(dog_data))])
    all_data_label = np.concatenate((cat_label[:sample_amount], dog_label))
    all_data = np.concatenate((sample_cat, dog_data))
    del sample_cat
    del dog_data
    gc.collect()
    split_limit = int(np.floor(0.7 * len(all_data)))
    random_index = random.sample(range((len(all_data))), split_limit)
    test_idx = set(np.arange(0, sample_amount)) - set(random_index)
    X_train = [all_data[i] for i in random_index]
    y_train = np.asarray([all_data_label[i] for i in random_index])
    X_test = [all_data[i] for i in test_idx]
    y_test = np.asarray([all_data_label[i] for i in test_idx])
    del cat_data
    gc.collect()
    return X_train, y_train, X_test, y_test


# In[10]:


cat_filenames = get_file_names('Cat')
dogtrain_filenames = get_file_names('DogTrain')
dogtest_filenames = get_file_names('DogTest')


# In[11]:


X_train, y_train, X_test, y_test = set_up_data(cat_filenames, dogtrain_filenames, dogtest_filenames,
                                              sample_amount=100)
num_classes = 2


# In[12]:


X_train = np.asarray(X_train).reshape(np.array(X_train).shape[0], 300, 300, 1)
X_test = np.asarray(X_test).reshape(np.array(X_test).shape[0], 300, 300, 1)


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# In[16]:


y_train.shape


# # Modeling

# In[17]:


print(X_train.shape, y_train.shape)
X_test.shape, y_test.shape


# In[18]:


# building a linear stack of layers with the sequential model
model = Sequential()
# hidden layer
model.add(Conv2D(25, kernel_size=(3,3), padding='valid',
                 activation='relu', input_shape=(300,300,1)))
# output layer
model.add(MaxPool2D(pool_size=(1,1)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(2, activation='softmax'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))


# In[22]:


model.predict(X_test)


# In[ ]:


model = Sequential()
# input_shape = (height, width, 1 if it's grayscale)
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(300,300,1), padding='same'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(2))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)


# In[ ]:


y_pred = model.predict(X_test)
y_pred


# In[ ]:


y_test


# In[ ]:


f1_score(y_pred, y_test)


# 0.51% accuracy for the first model.
