'''
参考：https://blog.csdn.net/shiheyingzhe/article/details/83062763#commentBox
使用letnet
'''

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import torch.nn as nn


# 加载数据, MNIST图片大小为28*28
def loadMNIST(batch_size):
    trans_img = transforms.Compose([transforms.ToTensor()])
    trainset = MNIST('./data',train=True,transform=trans_img,download=True)
    testset = MNIST('./data',train=False,transform=trans_img,download=True)
    trainload = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=10)
    testloader = DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=10)
    return trainset,testset,trainload,testloader


# 构建网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()   # 调用基类的__init__()函数
        '''torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
           stride=1, padding=0, dilation=1, groups=1, bias=True)
           
           torch.nn.MaxPool2d(kernel_size, stride=None, 
           padding=0, dilation=1, return_indices=False, ceil_mode=False)
        '''
        self.conv=nn.Sequential(  # 顺序网络结构
            nn.Conv2d(1, 6, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 120, 5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*120, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)     # 展平数据为7*7=49的一维向量
        out = self.fc(out)
        return out


# training
if __name__ == '__main__':
    learning_rate = 1e-2
    batch_size = 199
    epoches = 2
    gpus = [0]   # 使用哪几个GPU进行训练，这里选择0号GPU
    lenet = LeNet()   # 模型
    cuda_gpu = torch.cuda.is_available()    # 判断GPU是否存在可用
    if cuda_gpu:
        lenet = torch.nn.DataParallel(lenet, device_ids=gpus).cuda()
        print('USE GPU')
    else:
        print('USE CPU...')
    trainset, testset, trainloader, testloader = loadMNIST(batch_size)
    criterian = nn.CrossEntropyLoss(reduction='sum')   # 采用求和的方式计算loss
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)
    for i in range(epoches):
        running_loss = 0.0
        running_acc = 0.0
        for (img,label) in trainloader:
            optimizer.zero_grad()    # 求梯度之前对梯度清零以防止梯度累加
            output = lenet(img)    # 向前计算网络输出值
            loss = criterian(output, label)  # 计算loss
            loss.backward()
            optimizer.step()   # 使用计算好的梯度对参数进行更新

            running_loss += loss.item()
            valu, predict = torch.max(output, 1)  # 1的意思是按列求最大值，output的shape=torch.Size([100, 10])
                                                  # 其中torch.max()返回两个值，第一个是最大值，第二个是最大值的下标
            correct_num = (predict == label).sum()
            running_acc += correct_num.item()
        running_loss /= len(trainset)
        running_acc /= len(trainset)
        print("[%d/%d] Loss: %.5f, Acc: %.2f" % (i + 1, epoches, running_loss,
                                                 100 * running_acc))
