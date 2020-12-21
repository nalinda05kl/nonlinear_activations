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
alphas = 5
h_layers = 5
epochs = 20

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
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=4)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("\n Device using: ", device)

# pytorch implimentation of the network (only with dense layers)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        x = torch.rand(28,28).view(-1,1,28,28)
        self._to_linear = torch.flatten(x[0]).shape[0]
                
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.bn_1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 128)
        self.bn_3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, 128)
        self.bn_4 = nn.BatchNorm1d(128)
        
        self.fc5 = nn.Linear(128, 32)
        self.bn_5 = nn.BatchNorm1d(32)
        
        self.fc6 = nn.Linear(32, 10)
        
    def to_numpy(self, array, dim=32):
        '''convert torch tensor to numpy array'''
        array = array.to(device)
        return(array.view(dim).detach().numpy())
        
    def to_numpy_batch(self, in_batch, dim=32):
        '''convert a batch of torch tensors to a batch of numpy arrays'''
        in_batch = in_batch.to(device)
        out_batch = []
        for array in in_batch:
            out_batch.append(array.view(dim).detach().numpy())
        return np.array(out_batch)
        
    def entropy_calc(self, hgram, bw=1):
        '''entropy of per layer'''
        hgram = hgram.to(device)
        px = hgram/ float(np.sum(hgram))
        nzs = px > 0
        return -(np.sum(px[nzs] * np.log(px[nzs])) + np.log(1/bw))
            
    def entropy_batch_calc(self, array_in, dim=784, bins=32):
        '''average entropy per layer per batch'''
        array_in = array_in.to(device)
        entropy_cumulative = 0
        for image_in in array_in:
            hist_1d, x_edges = np.histogram(self.to_numpy(image_in, dim), bins=bins)
            entropy_cumulative += self.entropy_calc(hist_1d, bins)
        return entropy_cumulative/array_in.shape[0]
    
    def forward(self, x, entropy=True, alpha=0.01, bin_ratio=1):
        '''x;        input as batches
           entropy;  calculate entropy (in hidden layer output)
           alpha;    leaky relu non-linearity factor
        '''
        x = x.to(device)
        
        # ________________________ input ________________________
        x = x.view(-1, self._to_linear)
        
        # _________________________ HL1 _________________________
        x = F.leaky_relu(self.fc1(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim, int(dim/bin_ratio)
            H_batch_h1 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h1 = 0.
        # _________________________ HL2 _________________________ 
        x = F.leaky_relu(self.fc2(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim, int(dim/bin_ratio)
            H_batch_h2 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h2 = 0.       
        # _________________________ HL3 _________________________
        x = F.leaky_relu(self.fc3(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim, int(dim/bin_ratio)
            H_batch_h3 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h3 = 0.            
        # _________________________ HL4 _________________________
        x = F.leaky_relu(self.fc4(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim, int(dim/bin_ratio)
            H_batch_h4 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h4 = 0.            
        # _________________________ HL5 _________________________
        x = F.leaky_relu(self.fc5(x), alpha)
        dim = x.shape[1]
        if entropy:
            dim, bins = dim, int(dim/bin_ratio)
            H_batch_h5 = self.entropy_batch_calc(x, dim, bins)
        else:
            H_batch_h5 = 0.            
        # _________________________ HL6 _________________________
        x = F.relu(self.fc6(x))
        
        # _______________________ output ________________________
        x = F.log_softmax(x, dim=1)
                                
        return x, H_batch_h1, H_batch_h2, H_batch_h3, H_batch_h4, H_batch_h5
    
En_array = np.zeros((alphas, epochs, h_layers), dtype=np.float32).to(device)
perf_array = np.zeros((alphas, epochs, 4), dtype=np.float32).to(device)


for i in range(0,alphas):
    net = Net()
    # optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr = 0.001) # lr = 0.0001
    loss_function = nn.NLLLoss()
    
    alpha = 0.1*(i) # 0.01
    EPOCHS = epochs

    av_lossl, av_accl = [], []
    av_te_loss, av_te_acc = 0., 0.
    for epoch in tqdm(range(EPOCHS)):
        batches_count = 0
        total = 0
        num_correct = 0
        # training
        for batch_idx, (data, target) in enumerate(trainloader):
            batch_idx, data, target = batch_idx.to(device), data.to(device), target.to(device)
            if(batch_idx > 50):
                break
            batch_X, batch_y = data, target
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
            av_lossl.append(loss.detach().numpy())
            av_accl.append(float(num_correct)/float(total)*100)
        av_loss = np.array(av_lossl[-5:]).sum()/5.0
        av_acc = np.array(av_accl[-5:]).sum()/5.0 
        En_array[i][epoch]/= batches_count
        print(f"Epoch: {epoch}, Train Acc: {av_acc:.2f}, Train loss: {av_loss:.4f}") 
        print("H(x)", En_array[i][epoch])
        batches_count = 0
        num_correct = 0
        total = 0
        # switch to testing mode
        net.eval()
        with torch.no_grad():   
            for te_batch_idx, (te_data, te_target) in enumerate(testloader):
                te_batch_idx, te_data, te_target = te_batch_idx.to(device), te_data.to(device), te_target.to(device)
                if te_batch_idx > 20:
                    break
                te_batch_X, te_batch_y = te_data, te_target
                te_outputs = net.forward(te_batch_X, entropy=False, alpha=alpha)
                batches_count += 1
                _, pred = torch.max(te_outputs[0], 1)
                num_correct = (pred == te_batch_y).sum()
                total = te_batch_y.size(0)
                te_loss = loss_function(te_outputs[0], te_batch_y.long())
                av_te_loss += te_loss.detach().numpy()
                av_te_acc += float(num_correct)/float(total)*100
            av_te_loss /= batches_count
            av_te_acc /= batches_count
            print(f"Test Acc: {av_te_acc:.2f}, Test loss: {av_te_loss:4f}\n")
        # back to training mode
        net.train()
        perf_array[i][epoch][:] = av_acc, av_loss, av_te_acc, av_te_loss
    print ('[alpha = ', np.round(alpha,2), 'done]\n\n')
    
np.save("mnist_entropy.npy", En_array)
np.save("mnist_performance", perf_array)

print("\n....End....\n")
