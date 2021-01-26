import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data_path = "/Volumes/NalindaEXHard/Data"
batch_size = 100

transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]) #,
                               #torchvision.transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root=data_path, 
                                      train=True,
                                      download=True, 
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=4)
testset = torchvision.datasets.MNIST(root=data_path, 
                                     train=False,
                                     download=True, 
                                     transform=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n Device using: ", device)

# pytorch implimentation of the network (only with dense layers)
class Net(nn.Module):
    def __init__(self, alpha=0.0, init_weights=False):
        super(Net,self).__init__()
        
        self.alpha = alpha
                
        x = torch.rand(28,28).view(-1,1,28,28)
        self._to_linear = (torch.flatten(x[0]).shape[0])
                
        self.fc1 = nn.Linear(self._to_linear, 128)
        #self.bn_1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 512)
        #self.bn_2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 512)
        #self.bn_3 = nn.BatchNorm1d(512)
        
        self.fc4 = nn.Linear(512, 128)
        #self.bn_4 = nn.BatchNorm1d(128)
                
        self.fc5 = nn.Linear(128, 10)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        '''x;        input as batches
           alpha;    leaky relu non-linearity factor
        ''' 
        # ________________________ input ________________________
        x = x.view(-1, self._to_linear)
        x = x.to(device)
        # _________________________ HL1 _________________________
        x = F.leaky_relu(self.fc1(x), self.alpha)

        # _________________________ HL2 _________________________ 
        x = F.leaky_relu(self.fc2(x), self.alpha)
     
        # _________________________ HL3 _________________________
        x = F.leaky_relu(self.fc3(x), self.alpha)
           
        # _________________________ HL4 _________________________
        x = F.leaky_relu(self.fc4(x), self.alpha)
                
        # _______________________ output ________________________
        x = F.softmax(self.fc5(x), dim=1)
                                
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def performance_calc(net, dataloader, loss_calc='on', criterian=None, fast='off', limit=10):       
        num_correct = 0
        num_total = 0
        loss=0.
        with torch.no_grad():   
            for batch_idx_, (data_, target_) in enumerate(dataloader):
                batch_idx, data, target = batch_idx_, data_.to(device), target_.to(device)
                if fast == 'on':
                    if batch_idx > limit:
                        break
                batch_X, batch_y = data, target
                outputs = net(batch_X)
                pred = torch.argmax(outputs, axis=1)
                num_correct += (pred == batch_y).sum()
                num_total += batch_y.size(0)
                if loss_calc == 'on':
                    loss += criterian(outputs, batch_y.long())
        acc=(num_correct/num_total)*100.0
        loss/=(batch_idx+1)
        return acc.item(), loss.item()

epochs = 30 # 100
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
alphas_num  = len(alphas)
Perf_array = np.zeros((alphas_num, epochs, 4), dtype=np.float32)

acc_list = []
loss_list = []

for i in range(alphas_num):
    
    # alpha
    alpha = alphas[i]
    
    # initiate model
    net = Net(alpha=alpha).to(device)
        
    # optimizer and loss function
    optimizer = optim.SGD(net.parameters(), lr = 0.1)
    loss_function = nn.CrossEntropyLoss()
    
    EPOCHS = epochs
    
    print ("\n[alpha = ", alpha, " start]\n")

    for epoch in range(EPOCHS):
        batches_count = 0
        total = 0
        num_correct = 0
        
        # training
        for batch_idx, (data, target) in enumerate(trainloader):
            batch_idx, data, target = batch_idx, data.to(device), target.to(device)
            if(batch_idx > 400):
                break
            batch_X, batch_y = data.to(device), target.to(device)
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y.long())
            loss.backward()
            optimizer.step()

        # switch to testing mode
        net.eval()
        
        train_accuracy, train_loss = performance_calc(net, trainloader, criterian=loss_function, fast='on', limit=40)
        test_accuracy, test_loss = performance_calc(net, testloader, criterian=loss_function, fast='on', limit=200)
        
        # switching back to training mode
        net.train()
        
        Perf_array[i][epoch][:] = train_accuracy, train_loss, test_accuracy, test_loss
        
        # printing output
        # print(f"\nEpoch: {epoch}, \nTrain Acc: {train_accuracy:.2f}, Train loss: {train_loss:.4f}")
        if epoch == 29:
            print(f"Test Acc: {test_accuracy:.2f}, Test loss: {test_loss:4f}")

# pfa: performance function added
# np.save("performance_layers_5_alpha_0_1_pfa.npy", perf_array)

print("\n....End....\n")