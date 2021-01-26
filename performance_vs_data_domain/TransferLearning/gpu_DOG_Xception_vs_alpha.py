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

train_targets_path = "./Dog_Targets/dog_train_y.npy"
val_targets_path = "./Dog_Targets/dog_valid_y.npy"

# obtain one-hot vectors of labels (targets)
train_targets = np.load(train_targets_path)
valid_targets = np.load(val_targets_path)

# Obtain bottleneck features from pre-trained CNN.
model_name ="DogXception"
bn_ft_path = "./bottleneck_fts/DogXceptionData.npz"

bottleneck_features = np.load(bn_ft_path)
train_TL = bottleneck_features['train']
valid_TL = bottleneck_features['valid']
test_TL = bottleneck_features['test']

samples = 20 #20
alphas = 11
alpha_step = 0.1
learning_rate = 0.01
epochs = 30 #30
batch_size = 50
out_file_name = "./TL_Results/performance_"+ model_name + "_vs_alpha.npy"

# optimizer
SGD = optimizers.SGD(learning_rate)

Perf_array = np.zeros((samples, alphas, 4), dtype=np.float32)

for sample in range(samples):
    print("\nSample: ", sample+1)
    for al in range(alphas):
        alpha=al*0.1

        #if alpha!=0.1:
            #continue

        # model
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=train_TL.shape[1:]))
        model.add(Dense(128, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(512, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(512, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(128, activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(133, activation='softmax'))

        # compile the model
        model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])

        # fit
        history = model.fit(train_TL, train_targets, 
                  validation_data=(valid_TL, valid_targets),
                  epochs=epochs, 
                  batch_size=batch_size,
                  verbose=0)

        tr_acc  = np.amax(history.history['accuracy'])
        tr_loss = np.amin(history.history['loss'])
        val_acc = np.amax(history.history['val_accuracy'])
        val_loss= np.amin(history.history['val_loss'])
        Perf_array[sample][al][:] = tr_acc*100.0, tr_loss, val_acc*100.0, val_loss

        print(np.round(alpha, 2), ',\t', np.round(val_loss, 4), ',\t', np.round(val_acc*100.0, 4))
    
np.save(out_file_name, Perf_array)
print("\n      ... THE END ...      ")