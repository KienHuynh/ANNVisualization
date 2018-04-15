import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter

import numpy as np
import matplotlib.pyplot as plt


import pdb

class SimpleNet(nn.Module):
    def __init__(self, n_layer):
        super(SimpleNet, self).__init__()
        self.n_layer = n_layer
        self.linears = []
        for i in range(n_layer):
            self.linears.append(nn.Linear(2,2))
            setattr(self, 'linear' + str(i), self.linears[i])
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.elu = nn.ELU()
        self.nonlinear = F.tanh

    def forward(self, x):
        for i in range(self.n_layer-1):
            x = self.nonlinear(self.linears[i](x))
        x = self.linears[self.n_layer-1](x)
        return x


def generate_data(sample_size):
    """generate_data
    Generate a bunch of random data for 2 classes using Gaussian distribution
    :param sample_size: number of total samples (for both negative and positive classes)

    :return 
    """
    train_x = (np.random.randn(sample_size, 2)*0.125-0.15).astype(np.float32)
    train_y = np.zeros((sample_size, ), dtype=np.int)
    train_y[int(sample_size/2):] = 1
    train_x[train_y==1,:] += 0.3
    return train_x, train_y


def display_data(x, label, num_colors=1, color_index=0):
    color_map = np.asarray([
        [1, 0.5, 0],
        [0, 0.5, 1]
        ])
    color_map1 = np.asarray([
        [1,0,0],
        [0,0,1]
        ]) 
    color_map = color_map + color_index*(color_map1 - color_map)/float(num_colors)
    color_list = color_map[label, :]
    f1 = plt.figure(1)    
    plt.scatter(x[:,0], x[:,1], c=color_list, edgecolors=[0,0,0]) 
    plt.axis('equal')
    plt.grid(color=[0.5,0.5,0.5], linestyle='--')
    
    plt.show() 


def train(net, train_x, train_y):
    train_x = Variable(torch.from_numpy(train_x))
    train_y = Variable(torch.from_numpy(train_y))
    num_epoch = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    for e in range(num_epoch):
        optimizer.zero_grad()
        pred = net(train_x)
        display_data(pred.data.numpy(), train_y.data.numpy(), num_epoch, e+1)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()
        print('Epoch %03d: %.5f' % (e, loss))


def ff_visualization(net, train_x, train_y):
    x = Variable(torch.from_numpy(train_x))
    train_y = Variable(torch.from_numpy(train_y))
    for i, l in enumerate(net.linears[:-1]):
        #x = net.elu(l(x))
        x = net.nonlinear(l(x))
        display_data(x.data.numpy(), train_y.data.numpy(), len(net.linears), len(net.linears) - i)
    x = net.linears[-1](x)
    x = x.data.numpy()
    display_data(x, train_y.data.numpy(), len(net.linears), len(net.linears))


if __name__ == '__main__':
    np.random.seed(1311)
    torch.manual_seed(1311)
    train_x, train_y = generate_data(40)
    plt.hold(True)
    plt.ion()
    display_data(train_x, train_y)
    
    model = SimpleNet(5)
    train(model, train_x, train_y)
    #ff_visualization(model, train_x, train_y)
    pdb.set_trace()
