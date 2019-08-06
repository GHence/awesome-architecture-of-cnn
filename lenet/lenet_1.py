'''
下载MNIST数据集 测试LeNet网络模型
'''

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets    # 包含MNIST、COCO,LSUN CIFAR STL10等

# 定义超参数
batch_size = 128            # 批的大小
learning_rate = 1e-2        # 学习率
num_epoches = 20            # 遍历训练集的次数


# 数据类型转换，转换成numpy类型
def to_np(x):
    return x.cpu().data.numpy()


# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义 Convolution Network 模型
class Lenet(nn.Module):
    '''
    nn.Conv2d(in_dim, out_dim, Ksize, std, padding)
    '''
    def __init__(self, in_dim, n_class):
        super(Lenet, self).__init__()  # Lenet继承父类nn.Module的属性，并用父类的方法初始化这些属性
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 5, stride=1, padding=2),  # 由于MNIST图像大小为28*28，而LeNet输入为32*32，要使得数据大小和网络结构大小一致，
                                                           # 一般改网络大小而不是改数据的大小！
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),   # nn.Liner(in_feature, out_feature, bias=True)
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

        def forward(self, x):
            out = self.conv(x)
            out = out.view(out.size(0), -1)   # 相当于numpy中的resize()功能
            out = self.fc(out)
            return out


model = Lenet(1, 10)   # 图片大小为28*28，输入深度为1， 最终输出的10类
print('model = ',model)
use_gpu = torch.cuda.is_available()    # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
    print('USE GPU')
else:
    print('USE CPU')

# 定义损失函数loss和优化方式SGD
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# 训练模型
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch+1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        # cuda
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)   # 单次损失函数 * batchsize 作为每次循环的running_loss
        _, pred = torch.max(out, 1)   # 预测最大值所在的位置标签，即预测的数字
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()   # 清空所有被优化过的Variable的梯度
        loss.backward()         # 反向传播
        optimizer.step()        # 进行单次优化

    print('Finish {} epoch, Loss:{:.6f}, Acc:{:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))
    ))
    model.eval()    # 模型评估
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:      # 模型测试
        img, label = data
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(img, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss:{:.6f}. Acc:{:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))
    print()

# 保存模型
torch.save(model.state_dict(), './Lenet.pth')
