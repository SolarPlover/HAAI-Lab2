import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import BinarizeLinear
from torch.utils.data import DataLoader
import numpy as np
import os

# Training settings
use_cuda = torch.cuda.is_available()
batch_size=512                 
test_batch_size=2000	       
epochs=100
lr=0.01
seed=1                         # random seed (default: 1)
log_interval=30000 			   # how many batches to wait before logging training status



torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)


kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 1
        self.fc1 = BinarizeLinear(784, 64*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(64*self.infl_ratio, affine=False)
        self.fc2 = BinarizeLinear(64*self.infl_ratio, 64*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(64*self.infl_ratio, affine=False)
        # self.fc3 = BinarizeLinear(64*self.infl_ratio, 64*self.infl_ratio)
        # self.htanh3 = nn.Hardtanh()
        # self.bn3 = nn.BatchNorm1d(64*self.infl_ratio, affine=False)
        self.fc3 = nn.Linear(64*self.infl_ratio, 10)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        # x = self.fc3(x)
        # x = self.drop(x)
        # x = self.bn3(x)
        # x = self.htanh3(x)
        x = self.fc3(x)
        return x


model = Net()
if use_cuda:
    torch.cuda.set_device(0)
    model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        correct = 0
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            #  Print Accuracy
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            accuracy = 100. * torch.sum(correct) / len(output)
            accuracy_num = accuracy.item()
            print(f'Train Accuracy: {accuracy_num:.2f}%')
            
            #  Print LOSS
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(patience, best_loss):
	# !    FC bias=0   FC weights=1 or -1   No BatchNorm bias   No BatchNorm weights
    state_dict = model.state_dict()
    for key in state_dict:
        if key == 'fc1.weight' or key == 'fc2.weight' or key == 'fc3.weight' or key == 'fc3.weight' :
			# Get the parameter tensor
            param = state_dict[key]
			
			# Apply the threshold to convert values to 0 or 1
            param = torch.where(param >= 0.0, torch.tensor(1.0).cuda(), torch.tensor(-1.0).cuda()).cuda()
			
			# Update the state_dict with the new binary values
            state_dict[key] = param
        if key == 'fc1.bias' or key == 'fc2.bias' or key == 'fc3.bias' or key == 'fc3.bias' :
            param = state_dict[key]
            param = torch.where(param >= 0.0, torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()).cuda()
            state_dict[key] = param
            
        if key == 'bn1.bias' or key == 'bn2.bias' or key == 'bn3.bias':
            param = state_dict[key]
            param = torch.where(param >= 0.0, torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()).cuda()
            state_dict[key] = param
            

	# Load the modified state_dict back into the model
    model.load_state_dict(state_dict)
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if test_loss < best_loss:
        best_loss = test_loss
        patience = 5
        print(model)
        # print & save the model's parameters (state_dict)
        # check if model.txt exists, if exists, delete it
        if os.path.exists("model.txt"):
            os.remove("model.txt")
        # create a txt file and write the model's parameters (state_dict) to it
        f = open("model.txt", "a")       
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor])
            # write the model's parameters (state_dict) to the txt file
            if param_tensor == 'fc1.weight' or param_tensor == 'fc2.weight' or param_tensor == 'fc3.weight' :
                f.write(str(param_tensor) + "\n")
                tensor_numpy = model.state_dict()[param_tensor].cpu().numpy()
                np.savetxt(f, tensor_numpy, fmt='%s')
                f.write("\n")
        f.close()
        # dump the model
        torch.save(model, "model.pth")
    else:
        patience -= 1
		
    return patience, best_loss

if __name__ == '__main__':
    # patience = 5
    # best_loss = 1000000
    # for epoch in range(1, epochs + 1):
    #     train(epoch)
    #     patience, best_loss=test(patience, best_loss)
        
    #     if patience == 0:
    #         print('Early stopping...')
    #         break
    
    # use the trained model to predict the test set
    # load the trained model
    model = torch.load("model.pth")
    model.eval()
    with torch.no_grad():
        data7 =torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,148,210,253,253,113,87,148,55,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,87,232,252,253,189,210,252,252,253,168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,57,242,252,190,65,5,12,182,252,253,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,96,252,252,183,14,0,0,92,252,252,225,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,132,253,252,146,14,0,0,0,215,252,252,79,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,253,247,176,9,0,0,8,78,245,253,129,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,232,252,176,0,0,0,36,201,252,252,169,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,252,252,30,22,119,197,241,253,252,251,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,231,252,253,252,252,252,226,227,252,231,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,235,253,217,138,42,24,192,252,143,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,255,253,109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,71,253,252,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,253,252,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,71,253,252,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,106,253,252,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,255,253,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,218,252,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,96,252,189,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,184,252,170,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,147,252,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        data3 =torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 40, 103, 127, 127, 127, 127, 48, 40, 40, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 79, 79, 79, 84, 126, 126, 126, 126, 126, 126, 126, 126, 126, 105, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 113, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 120, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 126, 126, 126, 119, 56, 107, 126, 126, 126, 126, 126, 126, 126, 126, 126, 105, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 17, 17, 17, 15, 0, 15, 74, 17, 102, 117, 126, 126, 126, 126, 126, 118, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 0, 0, 0, 0, 0, 0, 0, 0, 17, 99, 126, 126, 126, 126, 122, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 16, 101, 101, 108, 126, 126, 126, 126, 120, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 83, 126, 126, 126, 126, 126, 126, 126, 119, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 126, 126, 126, 126, 126, 126, 126, 126, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 100, 126, 126, 126, 126, 126, 126, 126, 115, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 43, 43, 43, 124, 126, 126, 126, 126, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 76, 126, 126, 126, 125, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 119, 126, 126, 126, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 116, 126, 126, 75, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 101, 126, 126, 126, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 105, 105, 105, 29, 18, 18, 10, 13, 18, 75, 111, 126, 126, 126, 126, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 126, 126, 126, 126, 126, 126, 97, 107, 126, 126, 126, 126, 126, 126, 78, 38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 118, 117, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 78, 123, 126, 126, 126, 126, 126, 126, 126, 126, 79, 78, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58, 126, 126, 126, 126, 126, 63, 39, 39, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # convert the 1D array to 4D tensor
        data7 = data7.view(1, 1, 28, 28).float().cuda()
        data3 = data3.view(1, 1, 28, 28).float().cuda()
        data7 = Variable(data7)
        data3 = Variable(data3)
        output3 = model(data3)
        pred3 = output3.data.max(1, keepdim=True)[1]
        print(f'The predicted number is: {pred3.item()} while the actual number is: 3\n')
        output7 = model(data7)
        pred7 = output7.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        print(f'The predicted number is: {pred7.item()} while the actual number is: 7\n')
        
        
        