import numpy as np
from glob import glob
import os

from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from functools import partial
import tensorflow as tf
from tensorflow.keras import optimizers

# Obtain bottleneck features from pre-trained CNN. Select ONE!!
tr_bn_ft_path = "./bottleneck_fts/cifar10_VGG16_train_bn_ft.npz"
vl_bn_ft_path = "./bottleneck_fts/cifar10_VGG16_valid_bn_ft.npz"
tr_data_to_read = np.load(tr_bn_ft_path, mmap_mode='r')
vl_data_to_read = np.load(vl_bn_ft_path, mmap_mode='r')

train_TL = tr_data_to_read['features']
valid_TL = vl_data_to_read['features']

train_targets = np.load("./cifar10_Targets/y_train_cifar10.npy")
valid_targets = np.load("./cifar10_Targets/y_test_cifar10.npy")

samples = 2 #20
alphas = 11
alpha_step = 0.1
learning_rate = 0.01
epochs= 5 #30
batch_size= 50
model_name = "cifar10_VGG16"
out_file_name = "./TL_Results/performance_"+ model_name + "_vs_alpha.npy"

# optimizer
SGD = optimizers.SGD(learning_rate)

perf_array = np.zeros((samples, alphas, 4), dtype=np.float32)

for sample in range(samples):
    print("\nSample: ", sample)
    for al in range(alphas):
        alpha=al*0.1

        # model
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=train_TL.shape[1:]))
        model.add(Dense(128, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(512, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(512, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(128, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(10, activation='softmax'))

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])

        # fit
        history = model.fit(train_TL, train_targets, 
                  validation_data=(valid_TL, valid_targets),
                  epochs=epochs, :q!:
                  batch_size=batch_size,
                  verbose=0)

        tr_acc  = np.amax(history.history['accuracy'])
        tr_loss = np.amin(history.history['loss'])
        val_acc = np.amax(history.history['val_accuracy'])
        val_loss= np.amin(history.history['val_loss'])
        perf_array[sample][al][:] = tr_acc*100.0, tr_loss, val_acc*100.0, val_loss

        print(np.round(alpha, 2), ',\t', np.round(val_loss, 4), ',\t', np.round(val_acc*100.0, 4))
        
np.save(out_file_name, perf_array)
print("      ... THE END ...      ")
