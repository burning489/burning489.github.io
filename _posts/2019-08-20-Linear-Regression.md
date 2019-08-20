---
layout:     post
title:      Linear Regression
subtitle:   implement linear regression with PyTorch
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
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
```


```python
# hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 1e-3
```


```python
# set datasets
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# linear regression model
model = nn.Linear(input_size, output_size)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```


```python
# train the model
for epoch in range(num_epochs):
    # convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    
    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
```

    Epoch [5/60], Loss: 0.4896
    Epoch [10/60], Loss: 0.4888
    Epoch [15/60], Loss: 0.4879
    Epoch [20/60], Loss: 0.4871
    Epoch [25/60], Loss: 0.4863
    Epoch [30/60], Loss: 0.4855
    Epoch [35/60], Loss: 0.4847
    Epoch [40/60], Loss: 0.4838
    Epoch [45/60], Loss: 0.4830
    Epoch [50/60], Loss: 0.4822
    Epoch [55/60], Loss: 0.4814
    Epoch [60/60], Loss: 0.4806



![png](img/post-linear-regression.png)

