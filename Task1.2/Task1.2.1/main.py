import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F 

import torch.optim as optim

import nni 


transform = transforms.Compose(

    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)


trainset = torchvision.datasets.CIFAR10(root='./data',train = True,download = True,transform = transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

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


#--------------------训练过程--------------------

RECEIVED_PARAMS = nni.get_next_parameter()
    # 打印查看RECEIVED_PARAMS变量的内容和格式，也可方便debug


#print(RECEIVED_PARAMS)
    #将每次获取的参数记录到日志中
#logger.info("Received params:\n", RECEIVED_PARAMS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(),lr = RECEIVED_PARAMS['lr'],momentum = RECEIVED_PARAMS['momentum']) # 优化函数


for epoch in range(RECEIVED_PARAMS['epochs']):
    loss_running = 0.0
    for i,data in enumerate(trainloader,0):
        # get the inputs; data is a list of [inputs, labels]
        inputs,labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        # 计算loss
        loss = criterion(outputs,labels)
        # BP
        loss.backward()
        # optimize
        optimizer.step()

        #  打印
        loss_running +=loss.item()
        if i % 2000 ==1999:
            print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,loss_running /2000) )
            nni.report_intermediate_result(loss_running /2000)
            loss_running =0.0


print('finished Training')

# 保存训练模型 save to file 
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(),PATH)
# 从file中加载训练模型
# net.load_state_dict(torch.load(PATH)) 

#--------------- 测试过程--------------
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

nni.report_final_result( correct / total)
