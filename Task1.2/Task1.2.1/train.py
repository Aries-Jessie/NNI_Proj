import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import logging

import nni

_logger = logging.getLogger("cifar10_pytorch_automl")

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
         # nn.Linear(D_in,H) 
        # D_in:输入层 H:隐藏层
        self.fc1 = nn.Linear(16 * 5 * 5 ,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = view(-1, 16 * 5 * 5)
        x = x.view(x.size(0),-1) #reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def prepare(args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

    trainset = torchvision.datasets.CIFAR10 (root='./data',train = True,download = True,transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=args['batch_size'],shuffle=True,num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=args['batch_size'],shuffle=False,num_workers=2)

    net = Net().to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'])

    # if args['optimizer'] == 'SGD':
    #     optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
    # if args['optimizer'] == 'Adadelta':
    #     optimizer = optim.Adadelta(net.parameters(), lr=args['lr'])
    # if args['optimizer'] == 'Adagrad':
    #     optimizer = optim.Adagrad(net.parameters(), lr=args['lr'])
    # if args['optimizer'] == 'Adam':
    #     optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    # if args['optimizer'] == 'Adamax':
    #     optimizer = optim.Adam(net.parameters(), lr=args['lr'])


# Training
def train(args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    for epoch in range(args['epochs']):
        train_loss = 0.0
        for batch_idx,data in enumerate(trainloader,0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            # 计算loss
            loss = criterion(outputs,targets)
            # BP
            loss.backward()
            # optimize
            optimizer.step()

            #  打印
            train_loss +=loss.item()
            if batch_idx % 2000 ==1999:
                print('[%d,%5d] loss:%.3f'%(epoch+1,batch_idx+1,train_loss /2000) )
                nni.report_intermediate_result(train_loss /2000)
                train_loss =0.0

    print('Finished Training')
    return


# Testing
def test():
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    #net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            #loss = criterion(outputs, targets)

            #_, predicted = outputs.max(1)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
                   
    
    acc = 100.*correct/total
    print('Acc: %.3f%% (%d/%d)'% (acc, correct, total))
    
    return acc


if __name__ == '__main__':
  
    try:
        RCV_CONFIG = nni.get_next_parameter()
        #RCV_CONFIG = {'lr': 0.1, 'optimizer': 'Adam', 'model':'senet18'}
        _logger.debug(RCV_CONFIG)

        prepare(RCV_CONFIG)
        acc = 0.0
        train(RCV_CONFIG)
        acc= test()

        nni.report_final_result(acc)
    except Exception as exception:
        _logger.exception(exception)
        raise