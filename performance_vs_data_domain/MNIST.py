import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from functools import partial
from keras import optimizers
import os
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from __future__ import print_function

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_sample_size = 10000
test_sample_size = 100

x_train, y_train, x_test, y_test = x_train[:train_sample_size], y_train[:train_sample_size], x_test[:test_sample_size], y_test[:test_sample_size]

# Flatten the images to 1D array for the DNN model input
x_train = x_train.reshape(train_sample_size, 784)
x_test = x_test.reshape(test_sample_size, 784)
input_shape = (784, )

# Input shape and data type
input_shape = (784, )
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

# Optimizers for DNN
learning_rate = 0.1
SGD = optimizers.SGD(learning_rate)
ADAM = optimizers.Adam(learning_rate)

nodes1=128
nodes2=512
nodes3=512
nodes4=128

alpha_step = 0.1
steps = 11
samples = 20
epochs = 30
mean_sample_ratio = 0.2
mean_sample_index = int(epochs*mean_sample_ratio)

val_loss_list = []
val_loss_list_list = []


for sample in range(samples):
    val_loss_list = []
    for i in range(steps):
        alpha = i*alpha_step
        
        model = Sequential()
        model.add(Dense(nodes1, activation=partial(tf.nn.leaky_relu, alpha=alpha), input_shape=input_shape))
        model.add(Dense(nodes2, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(nodes3, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(nodes4, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(10, activation='softmax'))
    
        model.compile(optimizer=SGD,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
        history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=100, verbose=0)
    
        val_loss= np.mean(history.history['val_loss'][-mean_sample_index:])
        val_acc = np.mean(history.history['val_acc'][-mean_sample_index:])
        val_loss_list.append(val_loss)
    
        print(alpha, ',\t', val_loss, ',\t', val_acc)
    val_loss_list_list.append(val_loss_list)
    print('\n........ Sample [', sample, '] done ........\n')