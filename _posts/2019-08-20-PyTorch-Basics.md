---
layout:     post
title:      Pytorch Basics 
subtitle:   following tutorials about PyTorch
date:       2019-08-20
author:     Zhao Ding
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - PyTorch
---
> following tutorial from [github.yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/)


```python
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
```

### Basic autograd example1


```python
# create tensors
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# build a computational graph
y = w * x + b    # y = 2 * x + 3

# compute gradients / partial derivatives
y.backward()

# print out the gradients
print(x.grad, w.grad, b.grad)
```

    tensor(2.) tensor(1.) tensor(1.)


### Basic autograd example2


```python
# create tensors of shape (10, 3) and (10, 2) from normal ditribution
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# build a fully connected layer
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# build loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=1e-2)

# forward pass
pred = linear(x)

# compute loss
loss = criterion(pred, y)

# backward pass
loss.backward()

# print out the gradients
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# 1-step gradient descent
optimizer.step()

# print out the loss after 1-step gradient descent
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())
```

    w:  Parameter containing:
    tensor([[ 0.3928, -0.2780, -0.1483],
            [ 0.5744, -0.0531, -0.2090]], requires_grad=True)
    b:  Parameter containing:
    tensor([-0.1478, -0.1868], requires_grad=True)
    dL/dw:  tensor([[ 0.1492, -0.4448, -0.7934],
            [ 0.3848,  0.0321, -0.6033]])
    dL/db:  tensor([ 0.3319, -0.2593])
    loss after 1 step optimization:  1.5990203619003296


### load data from numpy


```python
# create a numpy array
x = np.array([[1, 2], [3, 4]])

# convert the numpy array to a torch tensor
y = torch.from_numpy(x)

# otherwise
z = y.numpy()
```

### input pipline


```python
# download and construct CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=False)
```


```python
# fetch one data pair (read data from disk)
image, label = train_dataset[0]
print(image.size())
print(label)
```

    torch.Size([3, 32, 32])
    6



```python
# data loader(provides queues and threads in a very simple way)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=64,
                                          shuffle=True)
# when itration starts, queue and thread start to load data from files
data_iter = iter(train_loader)

# Mini-batch images and labels
images, labels = data_iter.next()

# actual usage of the data loader is as below
for images, labels in train_loader:
    # training code should be written below
    pass
```

### input pipline for custom dataset


```python
# build custom dataset as below
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=False)
```

### pretrained model


```python
# Download and load the pretrained ResNet-18
resnet = torchvision.models.resnet18(pretrained=True)

# if want to finetune only the top layer
for param in resnet.parameters():
    param.requires_grad = False
    
# replace the top layer for finetuning
resnet.fc = nn.Linear(resnet.fc.in_features, 100)

# forward pass
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())
```

    torch.Size([64, 100])


### save and load model


```python
# save and load entire model
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# save and load only the model parameters ( recommended )
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
```




    <All keys matched successfully>


