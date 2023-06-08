import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
     ])

batch_size = 50

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)
train_set = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        # dropout
        self.dropout = lambda x: x
        # nn.Dropout(p=.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening
        x = x.view(-1, 64 * 4 * 4)
        # fully connected layers
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def get_default_device():
    if torch.cuda.is_available():
        print("GPU available")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()


net = Net().to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=0.02)


for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, (data, target) in enumerate(train_set):
        # zero the parameter gradients
        net.train()

        optimizer.zero_grad()
        data   = data.to(device)
        target = target.to(device)

        # forward + backward + optimize
        outputs = net(data)
        loss = criterion(outputs, target) 
        # loss = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

# print('Finished Training')
    net.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images   = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')





# (base) xidong@Lambda3:~$ python testCifar.py
# Files already downloaded and verified
# Files already downloaded and verified
# GPU available
# [1,   100] loss: 0.115
# [1,   200] loss: 0.115
# [1,   300] loss: 0.115
# [1,   400] loss: 0.114
# [1,   500] loss: 0.111
# [1,   600] loss: 0.106
# [1,   700] loss: 0.103
# [1,   800] loss: 0.099
# [1,   900] loss: 0.096
# [1,  1000] loss: 0.093
# Accuracy of the network on the 10000 test images: 35 %
# [2,   100] loss: 0.090
# [2,   200] loss: 0.090
# [2,   300] loss: 0.085
# [2,   400] loss: 0.085
# [2,   500] loss: 0.081
# [2,   600] loss: 0.081
# [2,   700] loss: 0.078
# [2,   800] loss: 0.078
# [2,   900] loss: 0.077
# [2,  1000] loss: 0.076
# Accuracy of the network on the 10000 test images: 43 %
# [3,   100] loss: 0.074
# [3,   200] loss: 0.074
# [3,   300] loss: 0.072
# [3,   400] loss: 0.071
# [3,   500] loss: 0.070
# [3,   600] loss: 0.070
# [3,   700] loss: 0.068
# [3,   800] loss: 0.066
# [3,   900] loss: 0.067
# [3,  1000] loss: 0.066
# Accuracy of the network on the 10000 test images: 53 %
# [4,   100] loss: 0.065
# [4,   200] loss: 0.064
# [4,   300] loss: 0.064
# [4,   400] loss: 0.063
# [4,   500] loss: 0.062
# [4,   600] loss: 0.061
# [4,   700] loss: 0.060
# [4,   800] loss: 0.061
# [4,   900] loss: 0.059
# [4,  1000] loss: 0.059
# Accuracy of the network on the 10000 test images: 58 %
# [5,   100] loss: 0.057
# [5,   200] loss: 0.057
# [5,   300] loss: 0.057
# [5,   400] loss: 0.056
# [5,   500] loss: 0.055
# [5,   600] loss: 0.054
# [5,   700] loss: 0.055
# [5,   800] loss: 0.054
# [5,   900] loss: 0.053
# [5,  1000] loss: 0.053
# Accuracy of the network on the 10000 test images: 63 %
# [6,   100] loss: 0.052
# [6,   200] loss: 0.050
# [6,   300] loss: 0.050
# [6,   400] loss: 0.051
# [6,   500] loss: 0.049
# [6,   600] loss: 0.050
# [6,   700] loss: 0.050
# [6,   800] loss: 0.049
# [6,   900] loss: 0.047
# [6,  1000] loss: 0.049
# Accuracy of the network on the 10000 test images: 63 %
# [7,   100] loss: 0.047
# [7,   200] loss: 0.045
# [7,   300] loss: 0.047
# [7,   400] loss: 0.044
# [7,   500] loss: 0.045
# [7,   600] loss: 0.045
# [7,   700] loss: 0.045
# [7,   800] loss: 0.045
# [7,   900] loss: 0.044
# [7,  1000] loss: 0.044
# Accuracy of the network on the 10000 test images: 66 %
# [8,   100] loss: 0.040
# [8,   200] loss: 0.041
# [8,   300] loss: 0.042
# [8,   400] loss: 0.041
# [8,   500] loss: 0.041
# [8,   600] loss: 0.040
# [8,   700] loss: 0.041
# [8,   800] loss: 0.041
# [8,   900] loss: 0.041
# [8,  1000] loss: 0.041
# Accuracy of the network on the 10000 test images: 68 %
# [9,   100] loss: 0.037
# [9,   200] loss: 0.036
# [9,   300] loss: 0.038
# [9,   400] loss: 0.038
# [9,   500] loss: 0.037
# [9,   600] loss: 0.037
# [9,   700] loss: 0.038
# [9,   800] loss: 0.037
# [9,   900] loss: 0.037
# [9,  1000] loss: 0.036
# Accuracy of the network on the 10000 test images: 69 %
# [10,   100] loss: 0.033
# [10,   200] loss: 0.033
# [10,   300] loss: 0.035
# [10,   400] loss: 0.034
# [10,   500] loss: 0.033
# [10,   600] loss: 0.034
# [10,   700] loss: 0.034
# [10,   800] loss: 0.033
# [10,   900] loss: 0.034
# [10,  1000] loss: 0.033
# Accuracy of the network on the 10000 test images: 71 %

