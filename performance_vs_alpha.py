import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms

import timeit

# for dictionary saving and reading
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
    
DATA_PATH = os.path.join(os.getcwd(), 'source_data')
#DATA_PATH = '/Volumes/NalindaEXHD/Data/'
TR_PATH = os.path.join(DATA_PATH, 'MNIST_train')
TE_PATH = os.path.join(DATA_PATH, 'MNIST_test')

if not os.path.exists(TR_PATH):
    os.makedirs(TR_PATH)
if not os.path.exists(TE_PATH):
    os.makedirs(TE_PATH)
        
train_path = os.path.abspath(TR_PATH)
test_path = os.path.abspath(TE_PATH)

transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.MNIST(train_path, download=True, train=True, transform=transform)
testset = datasets.MNIST(test_path, download=True, train=False, transform=transform)

# for sample testing
tr_size = 10000
te_size = 3000
sub_sample_trainset = th.utils.data.Subset(trainset, list(range(0, tr_size)))
sub_sample_testset = th.utils.data.Subset(testset, list(range(0, te_size)))
trainloader = th.utils.data.DataLoader(sub_sample_trainset, batch_size=100, shuffle=True)
testloader = th.utils.data.DataLoader(sub_sample_testset, batch_size=100, shuffle=True)

# for full run
#trainloader = th.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
#testloader = th.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# to check GPU
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("\n Device using: ", device)

input_size = 784
output_size = 10

class FCnetwork(nn.Module):
    def __init__(self, a_type='leaky_relu', nodes_list=[], non_line=0.1):
        super(FCnetwork, self).__init__()

        self.a_type = a_type

        if a_type == 'relu':
            self.activation = nn.ReLU()
        elif a_type == 'leaky_relu':
            self.activation = nn.LeakyReLU(non_line)
        else:
            print('activation not implemented')
            raise
            
        self.layer1 = th.nn.Sequential(*([th.nn.Linear(input_size, nodes_list[0]), self.activation]))
        self.layer2 = th.nn.Sequential(*([th.nn.Linear(nodes_list[0], nodes_list[1]), self.activation]))
        self.layer3 = th.nn.Sequential(*([th.nn.Linear(nodes_list[1], nodes_list[2]), self.activation]))
        self.classifier = nn.Sequential(*([nn.Linear(nodes_list[2], output_size)]))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = x.to(device)

        layer1 = self.layer1(x) 
        layer2 = self.layer2(layer1) 
        layer3 = self.layer3(layer2) 
        classifier = self.classifier(layer3)
                
        return classifier
        
    def accuracy(self, x, y):
        
        x = x.to(device)
        y = y.to(device)
        
        true = y
        x = x.view(x.shape[0], -1)
        classifier = self.forward(x)
        output = self.softmax(classifier)
        value, pred = th.max(output, 1)
        correct = 0
        count = 0
        for i, j in zip(true, pred):
            count+=1
            if int(i)==int(j):
                correct+=1
        return ((float(correct)/float(count))*100)
    
    
# Creating different model architectures
#nodes_list1 = [[100, 100, 100], [67, 134, 268], [116, 58, 29], [110, 55, 110], [87, 174, 87]] # shapes: =, >, <, <>, >< 
#nodes_list2 = [[16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128], [256, 256, 256]]    # shape : =
#nodes_list3 = [[16, 32, 64], [32, 64, 128], [64, 128, 256], [128, 256, 512], [32, 128, 512]]  # shape : <
#nodes_list4 = [[64, 32, 16], [128, 64, 32], [256, 128, 64], [512, 256, 128], [512, 128, 32]]  # shape : >
#nodes_list5 = [[32, 16, 32], [64, 32, 64], [128, 64, 128], [256, 128, 256], [256, 64, 256]]   # shape : ><
#nodes_list6 = [[16, 32, 16], [32, 64, 32], [64, 128, 64], [128, 256, 128], [64, 256, 64]]     # shape : <>

# testing
nodes_list1 =  [[100, 100, 100]]#, [67, 134, 268], [116, 58, 29], [110, 55, 110], [87, 174, 87]] 

nodes_lists =  nodes_list1

print('\n testing starts.\n')
start = timeit.default_timer()

epochs = 20
num_sub_samples = 1

for sample in range(0,num_sub_samples):
    
    acc_file_name = 'acc_'
    loss_file_name = 'loss_'
    val_acc_file_name = 'val_acc_'
    val_loss_file_name = 'val_loss_'
    
    acc_data_dict = {} # data for all network architectures. keys=archi, values=dicts of alpha vs acc
    loss_data_dict = {}
    val_acc_data_dict = {}
    val_loss_data_dict = {}
    
    for test in range(0,len(nodes_lists)):
        
        acc_data_dict_archi = {} # data per network architecture for all alpha. keys=alpha, values=acc
        loss_data_dict_archi = {}
        val_acc_data_dict_archi = {}
        val_loss_data_dict_archi = {}
        
        nodes_list = nodes_lists[test]
        print('\n___', nodes_list,'____')
        # Note: here the alpha goes form 0.0 to 'alpha_max' with 'num_steps' steps.
        alpha_max  = 0.2  # the maximum alpha (> 0.0 and <= 1.0)
        num_steps  = 2    # number of steps to increase the alpha to alpha_max
        alpha_step = alpha_max/float(num_steps)
        
        for step in range(0,num_steps):
            alpha = float(step*alpha_step)
            alpha = round(alpha, 2)
        
            model = FCnetwork('leaky_relu', nodes_list, alpha).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer =  th.optim.SGD(model.parameters(), lr=0.1)
    
            cost_list = []
            acc_list = []
            val_cost_list = []
            val_acc_list = []
    
            for epoch in range(epochs):
                acc = []
                for images, labels in trainloader:
                    images = images.view(images.shape[0], -1).to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    output = model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    with th.no_grad():
                        acc.append(model.accuracy(images, labels))
                acc_list.append(th.mean(th.tensor(acc), dim=0).tolist())   
                cost_list.append(loss.data)
                val_acc = []
                val_cost = []
                for v_images, v_labels in testloader:
                    v_images = v_images.to(device)
                    v_labels = v_labels.to(device)
                    with th.no_grad():
                        val_acc.append(model.accuracy(v_images, v_labels))
                        v_images = v_images.view(v_images.shape[0], -1)
                        val_output = model(Variable(v_images))
                        val_cost.append(criterion(val_output, Variable(v_labels)))
                        
                val_acc_list.append(th.mean(th.tensor(val_acc), dim=0).tolist()) 
                val_cost_list.append(th.mean(th.tensor(val_cost), dim=0).tolist())
        
                print('Run: {}'.format(sample+1),
                      '\talpha: {}'.format(round(alpha,2)), 
                      ',\tEN: {}'.format(epoch+1), 
                      '\t---> TL: {:.4f}'.format(cost_list[-1].item()), 
                      ',\tVL: {:.4f}'.format(val_cost_list[-1]), 
                      ',\tTA: {:.4f}'.format(acc_list[-1]), 
                      ',\tVA: {:.4f}'.format(val_acc_list[-1]))
            cost_list = [i.item() for i in cost_list]
            acc_data_dict_archi[str(alpha)] = acc_list
            loss_data_dict_archi[str(alpha)] = cost_list
            val_acc_data_dict_archi[str(alpha)] = val_acc_list
            val_loss_data_dict_archi[str(alpha)] = val_cost_list
            print('\n')
        acc_data_dict[str(nodes_lists[test])] = acc_data_dict_archi
        loss_data_dict[str(nodes_lists[test])] = loss_data_dict_archi
        val_acc_data_dict[str(nodes_lists[test])] = acc_data_dict_archi
        val_loss_data_dict[str(nodes_lists[test])] = loss_data_dict_archi
        
    acc_file_name += (str(sample) + '.p')
    acc_file_name = os.path.join('acc', acc_file_name)
    loss_file_name += (str(sample) + '.p')
    loss_file_name = os.path.join('loss', loss_file_name)
    val_acc_file_name += (str(sample) + '.p')
    val_acc_file_name = os.path.join('val_acc', val_acc_file_name)
    val_loss_file_name += (str(sample) + '.p')
    val_loss_file_name = os.path.join('val_loss', val_loss_file_name)
    
    #with open(acc_file_name, 'wb') as fp:
        #pickle.dump(acc_data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(loss_file_name, 'wb') as fp:
        #pickle.dump(loss_data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(val_acc_file_name, 'wb') as fp:
        #pickle.dump(val_acc_data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(val_loss_file_name, 'wb') as fp:
        #pickle.dump(val_loss_data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
               
stop = timeit.default_timer()
print('\n Time: ', stop - start)
print('\n testing done.\n')
