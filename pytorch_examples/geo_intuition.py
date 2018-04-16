import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

class SimpleNet(nn.Module):
    def __init__(self, n_layer, n_hidden, n_class):
        """__init__
        Init simple net. This is a simple feed forward neural network with (n_layer + 2) layers, each is fully connected.
        The input and output dims are both 2.
        The dim of the hidden layers are decided by n_hidden

        :param n_layer: int
        :param n_hidden: int
        :param n_class: int
        """
        super(SimpleNet, self).__init__()
        self.n_layer = n_layer + 2
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.linears = []
        self.linears.append(nn.Linear(2, n_hidden))
        setattr(self, 'linear0', self.linears[0])

        for i in range(1,self.n_layer-1):
            self.linears.append(nn.Linear(n_hidden, n_hidden))
            setattr(self, 'linear' + str(i), self.linears[i])

        if (num_class != 2):
            # For 2D visualization
            self.linears.append(nn.Linear(n_hidden, 2))
            setattr(self, 'linear' + str(self.n_layer-1), self.linears[self.n_layer-1])
            self.n_layer += 1
            
            self.linears.append(nn.Linear(2, 3))
            setattr(self, 'linear' + str(self.n_layer-1), self.linears[self.n_layer-1])


        else:
            self.linears.append(nn.Linear(n_hidden, n_class))
            setattr(self, 'linear' + str(self.n_layer-1), self.linears[self.n_layer-1])

        self.nonlinear = F.tanh
        #self.nonlinear = nn.LeakyReLU(0.1)
        self.nonlinear = nn.ELU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        for i in range(self.n_layer-1):
            if (i == self.n_layer-2 and self.n_class > 2):
                # This is so that the last layer is only a linear projection from this layer
                x = self.linears[i](x)
                break
            
            if (x.shape[0] == self.n_hidden):
                x = self.nonlinear(self.linears[i](x))
            else:
                x = self.nonlinear(self.linears[i](x)) 
        x_pre = x
        x = self.linears[self.n_layer-1](x)
        return x, x_pre


def generate_data_2class(sample_size):
    """generate_data_2class
    Generate a bunch of random data for 2 classes using Gaussian distribution
    :param sample_size: number of total samples (for both negative and positive classes)

    :return (train_x, train_y)
    """
    train_x = (np.random.randn(sample_size, 2)*0.125-0.1).astype(np.float32)
    train_y = np.zeros((sample_size, ), dtype=np.int)
    train_y[int(sample_size/2):] = 1
    train_x[train_y==1,:] += 0.2
    return train_x, train_y


def generate_data_3class(sample_size):
    """generate_data_3class
    Generate a bunch of random data for 3 classes using Gaussian distribution
    :param sample_size: number of total samples (for 3 classes)

    :return (train_x, train_y)
    """
    train_x = (np.random.randn(sample_size, 2)*0.125-0.1).astype(np.float32)
    train_y = np.zeros((sample_size, ), dtype=np.int)
    train_y[int(sample_size/3):int(2*sample_size/3)] = 1
    train_y[int(2*sample_size/3):] = 2
    train_x[train_y==1,:] += 0.2
    train_x[train_y==2,0] -= 0.2
    train_x[train_y==2,1] += 0.2

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


def display_data(x, label, num_colors=1, color_index=0, subplot_id=None, title='?', clear_fig=False, loss=0, num_class=2):
    if (num_class == 2):
        suptitle = 'Output of last layer at epoch %04d, train loss=%.5f' % (color_index, loss)
    else:
        suptitle = 'Output of second last layer at epoch %04d, train loss=%.5f' % (color_index, loss)

    use_color_map = True
    if (label.ndim == 2):
        if (label.shape[1] == 3):
            color_list = label
            use_color_map = False

    if use_color_map:
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
        plt.suptitle(suptitle)
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
    """display_motion
    This function draw the changes of locations of outputs over training iterations using lines
    It's very hard to see at the moment. I'm thinking of other methods to visualize this.

    :param data1: old points
    :param data2: new points
    :param e: current epoch
    :param num_e: total number of epochs
    """
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


def train(net, train_x, train_y, num_class):
    train_x = Variable(torch.from_numpy(train_x))
    train_y = Variable(torch.from_numpy(train_y))
   
    # Create grid data points
    #grid_points = generate_grid()
    min_x = -2
    max_x = 2
    min_y = -2
    max_y = 2
    x = np.linspace(min_x, max_x, 50)
    y = np.linspace(min_y, max_y, 50)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten().reshape(2500, 1)
    yv = yv.flatten().reshape(2500, 1)

    # Assign colors to these points
    grid_colors = np.zeros((50, 50, 3))
    grid_colors[:,:,0] = 1
    for i in range(50):
        grid_colors[i,:,0] -= i*1.0/50
        grid_colors[i,:,1] += i*1.0/50
        grid_colors[:,i,2] += i*1.0/50
    grid_colors = grid_colors.reshape((2500, 3))
    
    grid_points = np.concatenate((xv, yv), 1).astype(np.float32)
    grid_points = Variable(torch.from_numpy(grid_points)) 

    num_epoch = 800
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.01)

    old_grid_pred = None
    for e in range(num_epoch):
        optimizer.zero_grad()
        pred, pred_before_last = net(train_x)
        loss = criterion(pred, train_y)
        
        grid_pred, grid_pred_before_last = net(grid_points)
        if (num_class > 2):
            pred = pred_before_last
            grid_pred = grid_pred_before_last

        if (e % 100 == 0):   
            display_data(pred.data.numpy(), train_y.data.numpy(), num_epoch, e+1, 121, 'Using train data', clear_fig=True, loss=loss.data.numpy(), num_class=num_class)        
            display_data(grid_pred.data.numpy(), grid_colors, num_epoch, e+1, 122, 'Using points on a grid', loss=loss.data.numpy(), num_class=num_class) 
            #plt.savefig('./img/pytorch_examples/geo_intuition/video/%04d.jpg' % e)
            old_grid_pred = grid_pred.data.numpy()
           
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
    
    plt.hold(True)
    plt.ion()
    #train_x, train_y = generate_data_2class(10)
    #display_data(train_x, train_y)
    #num_class = 2
    #model = SimpleNet(5,5,num_class)
    #train(model, train_x, train_y)
   
    train_x, train_y = generate_data_3class(15)
    #display_data(train_x, train_y)
    num_class = 3
    model = SimpleNet(2,5,num_class)
    train(model, train_x, train_y, num_class)

    pdb.set_trace()
