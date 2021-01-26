import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from functools import partial
from tensorflow.keras import optimizers
from sklearn.datasets import make_regression

def make_exp_regression(n_samples, n_features, noice_std, shuffle=False):
    '''
    returns regression data in the form:
    y = exp(a_1*x_1 + a_2+x_2 + .... + a_n*x_n) + noice
    
    '''
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noice_std, shuffle=False)
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
learning_rate = 0.001 #0.0005
SGD = optimizers.SGD(learning_rate)

nodes=[128, 512, 512, 128]

alpha_step = 0.1
steps = 11
samples = 2 #20
epochs = 5 #50
mean_sample_ratio = 0.2
mean_sample_index = int(epochs*mean_sample_ratio)

val_loss_list_list = []
for sample in range(samples):
    val_loss_list = []
    for i in range(steps):
        alpha = i*alpha_step

        model = Sequential()
        model.add(Dense(nodes[0], activation=partial(tf.nn.leaky_relu, alpha=alpha), input_shape=input_shape))
        model.add(Dense(nodes[1], activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(nodes[2], activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(nodes[3], activation=partial(tf.nn.leaky_relu, alpha=alpha)))
        model.add(Dense(1))
    
        model.compile(optimizer=SGD,
                  loss='mean_squared_error',
                  metrics=['mse'])
    
        history = model.fit(x_train, y_train, validation_split=0.33, epochs=epochs, batch_size=n_samples, verbose=0)
    
        val_loss=np.mean(history.history['val_loss'][-mean_sample_index:])
        val_loss_list.append(val_loss)
        print(np.round(alpha, 2), ',\t', np.round(val_loss, 5))
    val_loss_list_list.append(val_loss_list)
    print('\n........ Sample [', sample, '] done ........\n')
    
val_loss_arr = np.array(val_loss_list_list)
val_loss_mean = np.mean(val_loss_arr, axis=0)
val_loss_std = np.std(val_loss_arr, axis=0)
val_loss_err = val_loss_std/np.sqrt(samples)
alphas = [np.round(i*0.1, 2) for i in range(steps)]

np.save("./RegResults/val_loss_mean_16ft.npy", val_loss_mean) 
np.save("./RegResults/val_loss_std_16ft.npy", val_loss_std)
np.save("./RegResults/val_loss_err_16ft.npy", val_loss_err)
np.save("./RegResults/alphas_16ft.npy", alphas)