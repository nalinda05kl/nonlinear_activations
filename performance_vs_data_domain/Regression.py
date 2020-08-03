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

def make_exp_regression(n_samples, n_features, noice, shuffle=False):
    '''
    returns regression data in the form:
    y = exp(a_1*x_1 + a_2+x_2 + .... + a_n*x_n) + noice
    
    '''
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=2, shuffle=False)
    y/= np.amax(y)
    y = np.exp(y)/np.amax(np.exp(y))
    X = X+4
    
    return X, y

n_samples  = 10000
n_features = 16
noice      = 1

X, y = make_exp_regression(n_samples, n_features, noice)

# Input shape and data type
input_shape = (n_features,)
x_train = X.astype('float32')
y_train = y.astype('float32')

# Optimizers for DNN
learning_rate = 0.0005
SGD = optimizers.SGD(learning_rate)

nodes1=128
nodes2=512
nodes3=512
nodes4=128

alpha_step = 0.1
steps = 11
samples = 20
epochs = 50
mean_sample_ratio = 0.2
mean_sample_index = int(epochs*mean_sample_ratio)

val_loss_list = []
val_loss_list_list = []


for sample in range(samples):
    for i in range(steps):
        alpha = i*alpha_step

        model = Sequential()
        model.add(Dense(nodes1, activation=partial(tf.nn.leaky_relu, alpha=alpha), input_shape=input_shape))
        model.add(Dense(nodes2, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(nodes3, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(nodes4, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(1))
    
        model.compile(optimizer=SGD,
                  loss='mean_squared_error',
                  metrics=['mse'])
    
        history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=n_samples, verbose=0)
    
        val_loss=np.mean(history.history['val_loss'][-mean_sample_index:])
        val_loss_list.append(val_loss)
    
        print(alpha, ',\t', val_loss)
    val_loss_list_list.append(val_loss_list)
    print('\n........ Sample [', sample, '] done ........\n')