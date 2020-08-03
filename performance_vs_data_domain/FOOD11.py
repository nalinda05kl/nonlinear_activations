# Classification of bottleneck features for FOOD-11 data (bottleneck features obtained using VGG-16)
# Contact: <nalida05kl@gmail.com> for bottleneck_features.npy, .hdf5 files and for other questions.
# Food 11 data : https://www.kaggle.com/vermaavi/food11

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Flatten, Dense, MaxPooling2D
from tensorflow.keras import applications
from tensorflow.keras import optimizers
import os
from tensorflow.keras.utils import to_categorical
import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint 
from functools import partial
from load_files import load_files
#from __future__ import print_function

## Change the file path according to the location of your data file.
train_data_dir = '/Volumes/NalindaEXHD/TSU-work/Data-food-11/train'
validation_data_dir = '/Volumes/NalindaEXHD/TSU-work/Data-food-11/valid'

## VGG16
top_model_weights_path = '/Volumes/NalindaEXHD/food-project-master/weights.best.VGG16.hdf5'

## InceptionV3
#top_model_weights_path = '/Volumes/NalindaEXHD/food-project-master/weights.best.InceptionV3.hdf5'

num_of_classes = 11
nb_train_samples = 3300 #4400
nb_validation_samples = 550 #660
epochs = 10
batch_size = 50

## define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path, shuffle=False, ignore_files=".DS_Store")
    food_targets = to_categorical(np.array(data['target']), num_of_classes)
    return food_targets

## load train, test, and validation datasets and extract the labels
train_labels = load_dataset(train_data_dir)
validation_labels = load_dataset(validation_data_dir)

## VGG16
train_data = np.load("/Volumes/NalindaEXHD/food-project-master/bottleneck_features_train_VGG16_V2_Tr3300_Val550.npy")
validation_data = np.load("/Volumes/NalindaEXHD/food-project-master/bottleneck_features_validation_VGG16_V2_Tr3300_Val550.npy")

## Check-point-best
checkpointer = ModelCheckpoint(filepath=top_model_weights_path, 
                               verbose=0, save_best_only=True)

# Optimizers for DNN
learning_rate = 0.01 #0.1
SGD = optimizers.SGD(learning_rate)
ADAM = optimizers.Adam(learning_rate)

alpha = 0.0
input_shape = train_data.shape[1:]
epochs = 20

nodes1=128
nodes2=512
nodes3=512
nodes4=128

alpha_step = 0.1
steps = 11 #11
samples = 2 # 20
mean_sample_ratio = 0.2
mean_sample_index = int(epochs*mean_sample_ratio)

val_loss_list = []
val_loss_list_list = []


for sample in range(samples):
    val_loss_list = []
    for i in range(steps):
        alpha = i*alpha_step
        
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=input_shape))
        model.add(Dense(nodes1, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(nodes2, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(nodes3, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(nodes4, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(11, activation='softmax'))
    
        model.compile(optimizer=SGD,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
        history = model.fit(train_data, 
                            train_labels, 
                            validation_data=(validation_data, validation_labels), 
                            epochs=epochs, 
                            batch_size=100, 
                            verbose=0)
    
        val_loss= np.mean(history.history['val_loss'][-mean_sample_index:])
        val_loss_list.append(val_loss)
    
        print('alpha: ', round(alpha,2), ',\t loss: ', val_loss)
    val_loss_list_list.append(val_loss_list)
    print('\n........ Sample [', sample+1, '] done ........\n')
print('\n<<<<<< END >>>>>>>')
