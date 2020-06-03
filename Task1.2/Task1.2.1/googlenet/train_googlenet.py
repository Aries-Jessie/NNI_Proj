import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import logging
import numpy as np
import PIL
from PIL import Image
import os

#from models import *
#from googlenet import *
from googlenet_V2A_SE import *


import nni

_logger = logging.getLogger("cifar10_googlenet_v3_automl")

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0


class Cutout(object):
    def __init__(self, hole_size):
        # 正方形马赛克的边长，像素为单位
        self.hole_size = hole_size

    def __call__(self, img):
        return cutout(img, self.hole_size)


def cutout(img, hole_size):
    y = np.random.randint(32)
    x = np.random.randint(32)

    half_size = hole_size // 2

    x1 = np.clip(x - half_size, 0, 32)
    x2 = np.clip(x + half_size, 0, 32)
    y1 = np.clip(y - half_size, 0, 32)
    y2 = np.clip(y + half_size, 0, 32)

    imgnp = np.array(img)

    imgnp[y1:y2, x1:x2] = 0
    img = Image.fromarray(imgnp.astype('uint8')).convert('RGB')
    return img


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

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        Cutout(6),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])



    trainset = torchvision.datasets.CIFAR10 (root='~/NNIProj/data',train = True,download = True,transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=args['batch_size'],shuffle=True,num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='~/NNIProj/data',train=False,download=True,transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,batch_size=args['batch_size'],shuffle=False,num_workers=2)

    # net = Net().to(device)
    net = GoogLeNet().to(device)

    # if args['model'] == 'googlenet':
    #     net = GoogLeNet()
    # if args['model'] == 'resnet34':
    #     net = ResNet34()
    # if args['model'] == 'senet18':
    #     net = SENet18()

    # net = net.to(device)
    if device == 'cuda':
        # 并行计算提高运行速度
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=5e-4)
    # optimizer = optim.Adadelta(net.parameters())

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
def train(epoch,args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    net.train()

    if epoch % args['threshold'] == 0:
        lr = args['lr']* 0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args['momentum'], weight_decay=5e-4)
    if epoch % 30 == 0:
        lr = args['lr']* 0.01
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args['momentum'], weight_decay=5e-4)
    if epoch % 40  == 0:
        lr = args['lr']* 0.001
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args['momentum'], weight_decay=5e-4)
    if epoch % 50  == 0:
        lr = args['lr']* 0.0005
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args['momentum'], weight_decay=5e-4)
    if epoch % 60 == 0:
        lr = args['lr']* 0.0001
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=args['momentum'], weight_decay=5e-4)
                   
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
            #nni.report_intermediate_result(train_loss/2000)
            train_loss =0.0

    print('Finished Training')
    return


# Testing
def test(epoch):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global best_acc

    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        test_running_loss = 0.0
        for data in testloader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            test_running_loss +=loss.item()
            
            #_, predicted = outputs.max(1)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
                   
    test_loss = 100*test_running_loss/total
    acc = 100.*correct/total
    print('Acc: %.3f%% (%d/%d)'% (acc, correct, total))
    if acc > best_acc:
        # print('Saving..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # checkpoint_dir = './checkpoint/'
        # dest_path = os.path.join(checkpoint_dir, "epoch_{}.pth".format(epoch))
        # torch.save(state, dest_path)
        best_acc = acc
    
    return acc,best_acc,test_loss


if __name__ == '__main__':
  
    try:
        RCV_CONFIG = nni.get_next_parameter()
        _logger.debug(RCV_CONFIG)

        prepare(RCV_CONFIG)
        acc = 0.0
        best_acc = 0.0
        # train(RCV_CONFIG)
        for epoch in range(RCV_CONFIG['epochs']):
            train(epoch,RCV_CONFIG)
            acc,best_acc,test_loss= test(epoch)
            nni.report_intermediate_result(test_loss)

        nni.report_final_result(acc)
    except Exception as exception:
        _logger.exception(exception)
        raise