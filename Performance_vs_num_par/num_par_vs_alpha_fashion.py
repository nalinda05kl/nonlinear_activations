import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import activations
from functools import partial

from matplotlib import pyplot as plt
from sklearn.datasets import make_regression

# MNIST digits data
#data_dict = np.load("./Data/mnist.npz")
#x_train, y_train = data_dict['x_train'], data_dict['y_train']
#x_test, y_test   = data_dict['x_test'], data_dict['y_test']

# MNIST fashion data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

train_sample_size = 60000 #60000
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
#ADAM = optimizers.Adam(learning_rate)

nodes_list1 = [[8, 8, 8], [16, 16, 16], [32, 32, 32], [64, 64, 64], [256, 256, 256]]
nodes_list2 = [[8, 16, 32], [16, 32, 64], [32, 64, 128], [64, 128, 256], [128, 256, 512]]
nodes_list3 = [[16, 8, 4], [32, 16, 8], [64, 32, 16], [128, 64, 32], [512, 256, 128]]
nodes_list4 = [[16, 8, 16], [32, 16, 32], [64, 32, 64], [128, 64, 128], [512, 256, 512]]
nodes_list5 = [[8, 16, 8], [16, 32, 16], [32, 64, 32], [64, 128, 64], [256, 512, 256]]
nodes_lists = nodes_list1 + nodes_list2 + nodes_list3 + nodes_list4 + nodes_list5

alpha_step_size = 0.1
alpha_steps = 11 #11
arch_shapes = len(nodes_lists)
samples = 20 #20
epochs = 20 #30

Perf_array = np.zeros((samples, arch_shapes, alpha_steps, 4), dtype=np.float32)

for sample in range(samples):
    print("\nSample [", sample, "] =======================>>")
    
    for arch in range(arch_shapes):
        print("\n........ ARC [", nodes_lists[arch], "] ........")
        
        nodes1, nodes2, nodes3 = nodes_lists[arch][0], nodes_lists[arch][1], nodes_lists[arch][2]
    
        for i in range(alpha_steps):

            alpha = i*alpha_step_size

            if i == 0:
                act = activations.relu
            else:
                act = partial(tf.nn.leaky_relu, alpha=alpha)

            model = Sequential()
            model.add(Dense(nodes1, activation=act, input_shape=input_shape))
            model.add(Dense(nodes2, activation=act))
            model.add(Dense(nodes3, activation=act))
            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer=SGD,
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])

            history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=64, verbose=0)

            tr_acc  = np.amax(history.history['accuracy'])
            tr_loss = np.amin(history.history['loss'])
            val_acc = np.amax(history.history['val_accuracy'])
            val_loss= np.amin(history.history['val_loss'])

            Perf_array[sample][arch][i][:] = tr_acc*100.0, tr_loss, val_acc*100.0, val_loss

            print(np.round(alpha, 2), ',\t', np.round(val_loss, 4), ',\t', np.round(val_acc*100.0, 4))
            
np.save("./par_alpha_Results/performance_fashion_num_par_vs_alpha.npy", Perf_array)
print("\n_______ END_______\n")
   
