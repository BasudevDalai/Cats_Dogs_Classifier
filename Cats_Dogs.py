# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:13:38 2019

@author: dalai
"""

# Building the CNN Architecture

# Importing the libraries
from keras.models import Sequential # To initialize the Neural Network as Neural Network is sequence of layers
from keras.layers import Conv2D # For convolution of layers
from keras.layers import MaxPooling2D # For pooling of layers
from keras.layers import Flatten # To convert the pooling feature map to a large feature vector
from keras.layers import Dense # To add the fully connected layers to ANN

# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')) # 32 feature maps, 3 by 3 size, 64 by 64 pixel images, 3 denoting color and relu used to increase non linearity

# MaxPooling
classifier.add(MaxPooling2D(pool_size = (2, 2))) # 2 by 2 maxpooling box size

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu')) # We don't specify the input shape as it was already mentioned previously and the input comes from the previous layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Fully connecting the layers
classifier.add(Dense(units = 128, activation = 'relu')) # 128 nodes in the hidden layer
classifier.add(Dense(units = 1, activation = 'sigmoid')) # Output layer (only one output as we have binary outcome, two outputs can also be used), since binary outcome so sigmoid function

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the recently created CNN architecture to images 

# Preprocessing the images to prevent overfitting (image augmentation)
from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Create training set
training_set = train_datagen.flow_from_directory('dataset/training_set', 
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating test/validation set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Fitting the CNN to training set and also checking its performance via test/validation set
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)



# Making predictions on new data

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'