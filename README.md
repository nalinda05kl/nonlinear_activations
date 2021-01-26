### Analysis Code for the Study: Effects of Nonlinearity and Network Architecture on the Performance of Supervised Neural Networks. 
Empirical study of the effects of nonlinearity in activation functions on the performance of deep learning models. We investigate the learning dynamics of hidden layers of shallow neural networks using entropy as a measurment of randomness in hidden layer outputs.

### This repository consists of following codes:
 - **Folder: Performance_vs_num_par**: The accuracy and loss calculated for different model architecturs using the MNIST data set as a function of linearity in the network. Both MNIST-digits and MNIST-fashion data sets were tested. Five model architecture shapes were tested by varing the number of parameters per layer (width).
 - **Folder: performance_vs_data_domain**: Loss calculated based on different data domains and with/without transfer learning.
   - Regression: Simulated data with 8, 16, 24 features
   - Classification (w/o TL): MNIST-fashion and MNIST-digits
   - Classification (w TL): FOOD-11, Dog Breeds, cifar10
     - TL pre-trained models: VGG16, VGG19, Xception, InceptionV3, ResNet50
 - **Folder: Entropy**: Entropy calculation for each layer and its variations under different nonlinearities and model architectures
 
 ### Data used for the analysis:
 - **MNIST hand written digits**: http://yann.lecun.com/exdb/mnist/
 - **FOOD-11 data set**: https://www.kaggle.com/vermaavi/food11
 - **DOG breeds data set**: 
 - **cifar10 data set**:
 - **MNIST fashion data set**:
 
 **Any questions please contact**: Nalinda Kulathunga (Nalinda.Kulathunga@tsu.edu)
