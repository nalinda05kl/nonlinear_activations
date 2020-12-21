import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data_path = "./Data"
batch_size = 64
h_layers = 6
epochs = 100
alpha_in = 0.1

transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                               (0.1307,), (0.3081,))])

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


archi_shapes = [[246, 246, 246, 246, 246, 246], [458, 229, 114, 57, 28, 14], [26, 52, 104, 208, 416, 832]]
#archi_shapes = [[246, 246, 246, 246, 246], [458, 229, 114, 57, 28], [26, 52, 104, 208, 416, 832]]
#archi_shapes = [[246, 246, 246, 246], [458, 229, 114, 57], [104, 208, 416, 832]]
#archi_shapes = [[246, 246, 246], [458, 229, 114], [208, 416, 832]]
#archi_shapes = [[246, 246], [458, 229], [416, 832]]

# pytorch implimentation of the network (only with dense layers)
class Net(nn.Module):
    def __init__(self, archi=archi_shapes[0], init_weights=True):
        super(Net,self).__init__()
        
        self.archi = archi
        
        x = torch.rand(28,28).view(-1,1,28,28)
        self._to_linear = (torch.flatten(x[0]).shape[0])
                
        self.fc1 = nn.Linear(self._to_linear, archi[0])
        self.bn_1 = nn.BatchNorm1d(archi[0])
        
        self.fc2 = nn.Linear(archi[0], archi[1])
        self.bn_2 = nn.BatchNorm1d(archi[1])
        
        self.fc3 = nn.Linear(archi[1], archi[2])
        self.bn_3 = nn.BatchNorm1d(archi[2])
        
        self.fc4 = nn.Linear(archi[2], archi[3])
        self.bn_4 = nn.BatchNorm1d(archi[3])
        
        self.fc5 = nn.Linear(archi[3], archi[4])
        self.bn_5 = nn.BatchNorm1d(archi[4])
        
        self.fc6 = nn.Linear(archi[4], archi[5])
        self.bn_6 = nn.BatchNorm1d(archi[5])
                
        self.fc7 = nn.Linear(archi[5], 10)
        if init_weights:
            self._initialize_weights()
        
    def to_numpy(self, array, dim=32):
        '''convert torch tensor to numpy array'''
        return(array.view(dim).detach().cpu().numpy())
        
    def to_numpy_batch(self, in_batch, dim=32):
        '''convert a batch of torch tensors to a batch of numpy arrays'''
        out_batch = []
        for array in in_batch:
            out_batch.append(array.view(dim).detach().cpu().numpy())
        return np.array(out_batch)
        
    def entropy_calc(self, hgram, bw=1):
        '''entropy of per layer (corrected)'''
        px = hgram/ float(np.sum(hgram))
        nzs = px > 0
        return -(np.sum(px[nzs] * np.log(px[nzs]))) + np.log(bw)
            
    def entropy_batch_calc(self, array_in, dim=784, bins=100):
        '''average entropy per layer per batch'''
        entropy_cumulative = 0
        for image_in in array_in:
            image_in_pos = image_in - torch.min(image_in)
            image_in_norm = image_in_pos/torch.max(image_in_pos)
            hist_1d, x_edges = np.histogram(self.to_numpy(image_in_norm, dim), bins=bins)
            entropy_cumulative += self.entropy_calc(hist_1d, bins)
        return entropy_cumulative/array_in.shape[0]
    
    def forward(self, x, entropy=True, alpha=0.1, bin_ratio=1):
        '''x;        input as batches
           entropy;  calculate entropy (in hidden layer output)
           alpha;    leaky relu non-linearity factor
        ''' 
        # ________________________ input ________________________
        x = x.view(-1, self._to_linear)
        x = x.to(device)
        # _________________________ HL1 _________________________
        x = F.leaky_relu(self.fc1(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim, 100 #int(dim/bin_ratio)
            H_batch_h1 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h1 = 0.
        # _________________________ HL2 _________________________ 
        x = F.leaky_relu(self.fc2(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim, 100 # int(dim/bin_ratio)
            H_batch_h2 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h2 = 0.       
        # _________________________ HL3 _________________________
        x = F.leaky_relu(self.fc3(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim, 100 # int(dim/bin_ratio)
            H_batch_h3 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h3 = 0.            
        # _________________________ HL4 _________________________
        x = F.leaky_relu(self.fc4(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim, 100 # int(dim/bin_ratio)
            H_batch_h4 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h4 = 0.            
        # _________________________ HL5 _________________________
        x = F.leaky_relu(self.fc5(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim,100 # int(dim/bin_ratio)
            H_batch_h5 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h5 = 0.
        # _________________________ HL6 _________________________
        x = F.leaky_relu(self.fc6(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim,100 # int(dim/bin_ratio)
            H_batch_h6 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h6 = 0. 
        # _________________________ HL6 _________________________
        x = F.relu(self.fc7(x))
        
        # _______________________ output ________________________
        x = F.log_softmax(x, dim=1)
                                
        return x, H_batch_h1, H_batch_h2, H_batch_h3, H_batch_h4, H_batch_h5, H_batch_h6

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
def performance_calc(model, dataloader, loss_calc='on', criterian=None, entropy=False, alpha=0.0, fast='off', limit=10):       
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
                outputs = net(batch_X, entropy=entropy, alpha=alpha)
                pred = torch.argmax(outputs[0], axis=1)
                num_correct += (pred == batch_y).sum()
                num_total += batch_y.size(0)
                if loss_calc == 'on':
                    loss += criterian(outputs[0], batch_y.long())
        acc=(num_correct/num_total)*100.0
        loss/=(batch_idx+1)
        return acc.item(), loss.item()
    
archi_num=len(archi_shapes)
En_array = np.zeros((archi_num, epochs, h_layers), dtype=np.float32)
perf_array = np.zeros((archi_num, epochs, 4), dtype=np.float32)

for i in range(archi_num):
    
    # initiate model
    net = Net(archi=archi_shapes[i]).to(device)
    
    # optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    loss_function = nn.NLLLoss()
    
    alpha = alpha_in 
    EPOCHS = epochs

    for epoch in range(EPOCHS):
        batches_count = 0
        total = 0
        num_correct = 0
        
        # training
        for batch_idx, (data, target) in enumerate(trainloader):
            batch_idx, data, target = batch_idx, data.to(device), target.to(device)
            #if(batch_idx > 10):
                #break
            batch_X, batch_y = data.to(device), target.to(device)
            net.zero_grad()
            outputs = net(batch_X, entropy=True, alpha=alpha)
            for hl in range(h_layers):
                En_array[i][epoch][hl] += outputs[hl+1]
            loss = loss_function(outputs[0], batch_y.long())
            loss.backward()
            optimizer.step()
            batches_count += 1
            _, pred = torch.max(outputs[0], 1)
            num_correct = (pred == batch_y).sum()
            total = batch_y.size(0)
        En_array[i][epoch]/= batches_count

        # switch to testing mode
        net.eval()

        train_accuracy, train_loss = performance_calc(net, trainloader, criterian=loss_function, alpha=alpha, fast='off')
        test_accuracy, test_loss = performance_calc(net, testloader, criterian=loss_function, alpha=alpha, fast='off')
        
        # switching back to training mode
        net.train()
        
        perf_array[i][epoch][:] = train_accuracy, train_loss, test_accuracy, test_loss
        
        # printing output
        print(f"\nEpoch: {epoch}, \nTrain Acc: {train_accuracy:.2f}, Train loss: {train_loss:.4f}")
        print(f"Test Acc: {test_accuracy:.2f}, Test loss: {test_loss:4f}")
        print("H(x)", En_array[i][epoch])
    print ("\n[archi = ", archi_shapes[i], "done]\n\n")

# pfa: performance function added
np.save("entropy_layers_6_alpha_0_1_pfa.npy", En_array)
np.save("performance_layers_6_alpha_0_1_pfa.npy", perf_array)

print("\n....End....\n")