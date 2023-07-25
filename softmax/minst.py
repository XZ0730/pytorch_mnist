import torch
import torchvision
from torch.utils.data import DataLoader
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
n_epochs = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100
random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_targets)
print(example_data.shape)

import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# model创建
class Net(nn.Module):
    def __init__(self):#继承nn并进行网络搭建
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 防止过拟合
        self.conv2_drop = nn.Dropout2d()
        # 全链接层
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)

        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        output = F.log_softmax(x)
        return output
    
network = Net().to(DEVICE)
#优化器定义  也有别的优化器例如Adam  作用是更新参数，权重
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

#损失值数组 用于存储所有的损失值
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #部署数据
        data ,target = data.to(DEVICE),target.to(DEVICE)
        #梯度归零，防止梯度累加
        optimizer.zero_grad()
        #模型训练获取输出
        output = network(data)
        #获取损失值
        loss = F.cross_entropy(output, target)
        #反向传播，更新参数
        loss.backward()
        #将更新的参数传播回去
        optimizer.step()
        #每6k4此进行一次打印
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
 
 #调用方法
train(1)
def test():
    # 模型的验证
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #部署数据
            data ,target = data.to(DEVICE),target.to(DEVICE)
            # 导入测试数据
            output = network(data)
            # 统计测试损失值
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            # 预测概率最大的结果的索引
            pred = output.data.max(1, keepdim=True)[1]
            # 统计正确率
            correct += pred.eq(target.data.view_as(pred)).sum()
    #获取损失值的平均值
    test_loss /= len(test_loader.dataset)
    #统计每轮测试的损失平均值
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 
test()
for epoch in range(1,n_epochs+1):
    train(epoch)
    test()


continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
 
network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)


for i in range(4, 9):
    test_counter.append(i*len(train_loader.dataset))
    train(i)
    test()
 
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
