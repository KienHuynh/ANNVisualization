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
        self.nonlinear = F.tanh
        self.softmax = nn.Softmax()

    def forward(self, x):
        for i in range(self.n_layer-1):
            x = self.nonlinear(self.linears[i](x)) + x
        x = self.linears[self.n_layer-1](x)
        return x


def generate_data(sample_size):
    """generate_data
    Generate a bunch of random data for 2 classes using Gaussian distribution
    :param sample_size: number of total samples (for both negative and positive classes)

    :return 
    """
    train_x = (np.random.randn(sample_size, 2)*0.125-0.1).astype(np.float32)
    train_y = np.zeros((sample_size, ), dtype=np.int)
    train_y[int(sample_size/2):] = 1
    train_x[train_y==1,:] += 0.2
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
    plt.grid(color=[0.5,0.5,0.5], linestyle='--') 
    plt.show()
    plt.pause(0.05)


global_field = None
def color_space(net, x):
    global global_field
    color_map = np.asarray([
        [1, 0.8, 0.2],
        [0.2, 0.8, 1]
        ])
    #min_x = np.min(x[:,0])
    #min_y = np.min(x[:,1])
    #max_x = np.max(x[:,0])
    #max_y = np.max(x[:,1])
    min_x = -10
    max_x = 10
    min_y = -10
    max_y = 10
    x = np.linspace(min_x, max_x, 40)
    y = np.linspace(min_y, max_y, 40)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten().reshape(1600, 1)
    yv = yv.flatten().reshape(1600, 1)
    xy = np.concatenate((xv, yv), 1).astype(np.float32)
    xy = Variable(torch.from_numpy(xy))
    #pdb.set_trace()
    
    pred = net.softmax(net(xy))
    pred = pred.data.numpy()
    pred[pred[:,0]>0.5,:] = 0
    pred[pred[:,0] != 0,:] = 1
    pred = pred[:,0].astype(np.int)
    color_list = color_map[pred,:]
    if global_field is not None:
        global_field.remove()
        plt.draw()
    global_field = plt.scatter(xv[:], yv[:], c=color_list)

def train(net, train_x, train_y):
    train_x = Variable(torch.from_numpy(train_x))
    train_y = Variable(torch.from_numpy(train_y))
    num_epoch = 400
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    for e in range(num_epoch):
        optimizer.zero_grad()
        pred = net(train_x)
        if (e%4==0):
            color_space(net, pred.data.numpy())
            display_data(pred.data.numpy(), train_y.data.numpy(), num_epoch, e+1)
            
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()
        print('Epoch %03d: %.5f' % (e, loss))


def ff_visualization(net, train_x, train_y):
    x = Variable(torch.from_numpy(train_x))
    train_y = Variable(torch.from_numpy(train_y))
    for i, l in enumerate(net.linears[:-1]):
        x = net.nonlinear(l(x))
        display_data(x.data.numpy(), train_y.data.numpy(), len(net.linears), len(net.linears) - i)
    x = net.linears[-1](x)
    x = x.data.numpy()
    display_data(x, train_y.data.numpy(), len(net.linears), len(net.linears))


if __name__ == '__main__':
    np.random.seed(1311)
    torch.manual_seed(1311)
    train_x, train_y = generate_data(10)
    plt.hold(True)
    plt.ion()
    #display_data(train_x, train_y)
    
    model = SimpleNet(5)
    train(model, train_x, train_y)
    plt.axis('equal')
    pdb.set_trace()
