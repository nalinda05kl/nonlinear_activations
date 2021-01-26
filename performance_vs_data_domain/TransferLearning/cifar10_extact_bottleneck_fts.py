import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D

# Load the dataset:
(X_trainf, y_trainf), (X_testf, y_testf) = cifar100.load_data()

# Using 10000 images
X_train, y_train, X_test, y_test = X_trainf[:9000], y_trainf[:9000], X_testf[:1000], y_testf[:1000]

# One-hot encoding the labels
num_classes = 10
from keras.utils import np_utils
y_train_oh = np_utils.to_categorical(y_train, num_classes)
y_test_oh = np_utils.to_categorical(y_test, num_classes)

# Save lables
np.save("y_train_cifar10.npy", y_train_oh)
np.save("y_test_cifar10.npy", y_test_oh)

y_train = np.load("y_train_cifar10.npy")
y_test = np.load("y_test_cifar10.npy")

#Importing the ResNet50 or VGG16 model
#from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input

#Loading the ResNet50 model with pre-trained ImageNet weights
#model_name = "cifar10_ResNet50"
#model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

model_name = "cifar10_VGG16"
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Reshaping the training data
import tensorflow as tf
X_train_new = np.array([tf.image.resize(X_train[i], [224, 224], method=tf.image.ResizeMethod.BILINEAR) for i in range(0,len(X_train))]).astype('float32')

#Preprocessing the data, so that it can be fed to the pre-trained ResNet50 model. 
resnet_train_input = preprocess_input(X_train_new)

#Creating bottleneck features for the training data
train_features = model.predict(resnet_train_input)

# Saving the bottleneck features
out_file_name = model_name + "_train_bn_ft"
np.savez(out_file_name, features=train_features)

X_test_new = np.array([tf.image.resize(X_test[i], [224, 224], method=tf.image.ResizeMethod.BILINEAR) for i in range(0,len(X_test))]).astype('float32')

resnet_test_input = preprocess_input(X_test_new)

#Creating bottleneck features for the testing data
test_features = model.predict(resnet_test_input)

#Saving the bottleneck features
outv_file_name = model_name + "_valid_bn_ft"
np.savez(outv_file_name, features=test_features)
