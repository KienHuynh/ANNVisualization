import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter

import numpy as np
import matplotlib.pyplot as plt


class SimpleNet(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 2)
        self.linear4 = nn.Linear(2, 2)
        self.linear5 = nn.Linear(2, 2)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.leaky_relu(self.linear3(x))
        x = self.leaky_relu(self.linear4(x))
        x = self.linear5(x)
        return x


def generate_data(sample_size):
    """generate_data
    Generate a bunch of random data for 2 classes using Gaussian distribution
    :param sample_size: number of total samples (for both negative and positive classes)

    :return 
    """
    train_x = np.random.randn(sample_size, 2)
    train_y = np.zeros((sample_size, ), dtype=np.int)
    train_y[int(sample_size/2):] = 1
    return train_x, train_y


def display_data(x, label):
    color_map = np.asarray([
        [1, 0.8, 0],
        [0, 0, 1]
        ])
    
    color_list = color_map[label, :]
    plt.scatter(x[:,0], x[:,1], c=color_list, edgecolors=[0,0,0]) 
    plt.show()

if __name__ == '__main__':
    np.random.seed(1311)
    torch.manual_seed(1311)
    train_x, train_y = generate_data(40)
    display_data(train_x, train_y)
    

