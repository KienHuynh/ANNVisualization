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
    def __init__(self, n_layer, n_hidden):
        """__init__
        Init simple net. This is a simple feed forward neural network with (n_layer + 2) layers, each is fully connected.
        The input and output dims are both 2.
        The dim of the hidden layers are decided by n_hidden

        :param n_layer: int
        :param n_hidden: int
        """
        super(SimpleNet, self).__init__()
        self.n_layer = n_layer + 2
        self.n_hidden = n_hidden
        self.linears = []
        self.linears.append(nn.Linear(2,n_hidden))
        setattr(self, 'linear0', self.linears[0])

        for i in range(1,self.n_layer-1):
            self.linears.append(nn.Linear(n_hidden, n_hidden))
            setattr(self, 'linear' + str(i), self.linears[i])

        self.linears.append(nn.Linear(n_hidden,2))
        setattr(self, 'linear' + str(self.n_layer-1), self.linears[self.n_layer-1])

        self.nonlinear = F.tanh
        #self.nonlinear = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        for i in range(self.n_layer-1):
            if (x.shape[0] == self.n_hidden):
                x = self.nonlinear(self.linears[i](x)) + x
            else:
                x = self.nonlinear(self.linears[i](x))
        x = self.linears[self.n_layer-1](x)
        return x


def generate_data(sample_size):
    """generate_data
    Generate a bunch of random data for 2 classes using Gaussian distribution
    :param sample_size: number of total samples (for both negative and positive classes)

    :return (train_x, train_y)
    """
    train_x = (np.random.randn(sample_size, 2)*0.125-0.1).astype(np.float32)
    train_y = np.zeros((sample_size, ), dtype=np.int)
    train_y[int(sample_size/2):] = 1
    train_x[train_y==1,:] += 0.2
    return train_x, train_y


def generate_grid():
    """generate_grid
    Generate points over a grid, note that this is different from np.meshgrid
    """

    min_x = -10
    max_x = 10
    min_y = -10
    max_y = 10
    num_line = 10
    x = np.linspace(min_x, max_x, num_line).reshape(num_line,1)
    y = np.linspace(min_y, max_y, num_line).reshape(num_line,1)
    xy = np.asarray([]).reshape((0,2))
    for i in range(min_x, max_x):    
        xi = np.ones_like(y)*i
        xi = np.concatenate((xi, y), 1)
        xv = np.concatenate((xy, xi), 0)
    
    for i in range(min_y, max_y):    
        yi = np.ones_like(x)*i
        yi = np.concatenate((x, yi), 1)
        xy = np.concatenate((xy, yi), 0)

    return xy.astype(np.float32)


def display_data(x, label, num_colors=1, color_index=0, subplot_id=None, title='?', clear_fig=False):
    color_map = np.asarray([
        [1, 0.5, 0],
        [0, 0.5, 1],
        [0.5, 1, 0],
        ])
    color_map1 = np.asarray([
        [1,0,0],
        [0,0,1],
        [0,1,0]
        ]) 
    color_map = color_map + color_index*(color_map1 - color_map)/float(num_colors)
    color_list = color_map[label, :]

     
    if (subplot_id == None):
        f = plt.figure(1, figsize=(7,7), dpi=80, facecolor='w', edgecolor='k')      
        if (clear_fig):
            f.clear()
    else:
        f = plt.figure(num=2, figsize=(14, 7), dpi=80, facecolor='w', edgecolor='k')  
        if (clear_fig):
            f.clear()
        axarr = plt.subplot(subplot_id)
        axarr.set_title(title)
    
    

    plt.scatter(x[:,0], x[:,1], c=color_list, edgecolors=[0,0,0]) 
    plt.grid(color=[0.5,0.5,0.5], linestyle='--') 
    plt.show()
    plt.axis('equal')
    plt.pause(0.05)
    global_field = None


def display_motion(data1, data2, e, num_e):
    color_map = np.asarray([0, 0.5, 1])
    color_map1 = np.asarray([0,0,1]) 
    color_map = color_map + e*(color_map1 - color_map)/float(num_e)

   
    f3 = plt.figure(3, figsize=(7,7), dpi=80, facecolor='w', edgecolor='k')      
    plt.hold(True)
    
    for i in range(data1.shape[0]):
        plt.plot([data1[i][0], data2[i][0]], [data1[i][1], data2[i][1]], c=color_map)

    plt.grid(color=[0.5,0.5,0.5], linestyle='--') 
    plt.show()
    plt.axis('equal')
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
    x = np.linspace(min_x, max_x, 10)
    y = np.linspace(min_y, max_y, 10)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten().reshape(1600, 1)
    yv = yv.flatten().reshape(1600, 1)
    xy = np.concatenate((xv, yv), 1).astype(np.float32)
    xy = Variable(torch.from_numpy(xy))
    #pdb.set_trace()
    
    pred = net.softmax(net(xy))
    pred = pred.data.numpy()
    pred[pred[:,0]>0.5,:] = 1
    pred[pred[:,0] != 1,:] = 0
    pred = pred[:,0].astype(np.int)
    color_list = color_map[pred,:]
    if global_field is not None:
        global_field.remove()
        plt.draw()
    global_field = plt.scatter(xv[:], yv[:], c=color_list)


def train(net, train_x, train_y):
    train_x = Variable(torch.from_numpy(train_x))
    train_y = Variable(torch.from_numpy(train_y))
    
    grid_points = generate_grid()
    grid_label = (np.ones_like(grid_points[:,0])*2).astype(np.int)
    grid_points = Variable(torch.from_numpy(grid_points))


    num_epoch = 600
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    old_grid_pred = None
    for e in range(num_epoch):
        optimizer.zero_grad()
        pred = net(train_x)
        if (e%4==0):
            #color_space(net, pred.data.numpy())
            pred_y = net.softmax(pred).data.numpy()

            pred_y[pred_y[:,0]>0.5,:] = 0
            pred_y[pred_y[:,0] != 0, :] = 1
            pred_y = pred_y[:,0]
            display_data(pred.data.numpy(), pred_y.astype(np.int), num_epoch, e+1, 121, 'Predictions', clear_fig=True)
            
            #display_data(pred.data.numpy(), train_y.data.numpy(), num_epoch, e+1, 122, 'Labels')

            grid_pred = net(grid_points)
            display_data(grid_pred.data.numpy(), grid_label, num_epoch, e+1, 122, 'Labels')
           
            if (old_grid_pred is not None):
                display_motion(old_grid_pred, grid_pred, e+1, num_epoch)

            old_grid_pred = grid_pred.data.numpy()
            #display_data(pred.data.numpy(), train_y.data.numpy(), num_epoch, e+1,clear_fig=True)
   
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
    
    model = SimpleNet(3,5)
    train(model, train_x, train_y)
    
    pdb.set_trace()
