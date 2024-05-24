import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import BinarizeLinear
from torch.utils.data import DataLoader

# Training settings
use_cuda = torch.cuda.is_available()
batch_size=512                 # input batch size for training (default: 64)
test_batch_size=2000	       # input batch size for testing (default: 1000)
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
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio, affine=False)
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio, affine=False)
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio, affine=False)
        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        # self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        # return self.logsoftmax(x)
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

        # if epoch % 40 == 0:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))

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
        if key == 'fc1.weight' or key == 'fc2.weight' or key == 'fc3.weight' or key == 'fc4.weight' :
			# Get the parameter tensor
            param = state_dict[key]
			# print(param)
			
			# Apply the threshold to convert values to 0 or 1
            param = torch.where(param >= 0.0, torch.tensor(1.0).cuda(), torch.tensor(-1.0).cuda()).cuda()
			
			# Update the state_dict with the new binary values
            state_dict[key] = param
        if key == 'fc1.bias' or key == 'fc2.bias' or key == 'fc3.bias' or key == 'fc4.bias' :
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
        # dump the model
        torch.save(model.state_dict(), 'model.pth')
        print(model)
        # print the model's parameters (state_dict)

        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor])
		# dump the optimizer
        # torch.save(optimizer.state_dict(), 'optimizer.pth')
    else:
        patience -= 1
		
    return patience, best_loss

if __name__ == '__main__':
    patience = 5
    best_loss = 1000000
    for epoch in range(1, epochs + 1):
        train(epoch)
        patience, best_loss=test(patience, best_loss)
        
        # if epoch % 40 == 0:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
        if patience == 0:
            print('Early stopping...')
            break