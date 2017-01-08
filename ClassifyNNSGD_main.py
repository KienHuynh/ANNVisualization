import numpy as np
import matplotlib.pyplot as plt
import pylab
import time
import matplotlib.text as plttext
from SGDClassifyNNDemo import *
# This guy is global

# Create a 2D dataset with 3 labels
def create_data(num_sample=None):
    """
    Generate samples of 3 classes using normal distribution

    :type num_sample: int
    :param num_sample: number of sample to be generated for each class

    :return: data and label for each class
    """
    I = np.eye(3, dtype=np.float32)


    if (num_sample == None):
        num_sample = 100

    # Generate first class
    m1 = np.asarray([0, 0], dtype=np.float32)
    cov1 = np.asarray([[0.5, 0], [0, 0.5]], dtype=np.float32)
    data1 = rng.multivariate_normal(m1, cov1, num_sample)
    label1 = np.ones((num_sample), dtype=np.uint16) - 1
    label1 = I[label1,:]

    # Generate second class
    m2 = np.asarray([5,5], dtype=np.float32)
    cov2 = np.asarray([[0.5, 0], [0, 0.5]], dtype=np.float32)
    data2 = rng.multivariate_normal(m2, cov2, num_sample)
    label2 = np.ones((num_sample), dtype=np.uint16)
    label2 = I[label2, :]

    # Generate third class
    noise = np.abs((np.reshape(rng.normal(0, 0.01, num_sample), (num_sample,1))))
    S1 = np.asarray([[1, 0], [0, 0.7]], dtype=np.float32)
    S2 = np.asarray([[4, 0], [0, 4]], dtype=np.float32)
    m3 = np.asarray([0.5, 0.5], dtype=np.float32)
    cov3 = np.asarray([[0.5, 0], [0, 0.5]], dtype=np.float32)
    data3 = rng.multivariate_normal(m3, cov3, num_sample)
    data3 = data3/np.repeat(np.sqrt(np.sum(data3**2, 1, keepdims=True) + noise), 2, 1)
    data3 = np.dot(S2, np.dot(S1, data3.T)).T

    d = np.sqrt(np.sum(data3**2, 1, keepdims=True))
    d1 = np.reshape(d<2.5, (num_sample))
    data3[np.ix_(d1, [True, True])] = data3[np.ix_(d1, [True, True])]/np.repeat(d[d1], 2, 1)

    label3 = np.ones((num_sample), dtype=np.uint16) + 1
    label3 = I[label3, :]

    return (data1, label1, data2, label2, data3, label3)

def create_train_val_test(data1, label1, data2, label2, data3, label3, num_train, num_val):
    train_X = np.concatenate((data1[0:num_train, :],
                              data2[0:num_train, :],
                              data3[0:num_train, :]))

    train_Y = np.concatenate((label1[0:num_train, :],
                              label2[0:num_train, :],
                              label3[0:num_train, :]))

    val_X = np.concatenate((data1[num_train:(num_train+num_val), :],
                              data2[num_train:(num_train+num_val), :],
                              data3[num_train:(num_train+num_val), :]))

    val_Y = np.concatenate((label1[num_train:(num_train+num_val), :],
                              label2[num_train:(num_train+num_val), :],
                              label3[num_train:(num_train+num_val), :]))

    test_X = np.concatenate((data1[(num_train + num_val):, :],
                            data2[(num_train + num_val):, :],
                            data3[(num_train + num_val):, :]))

    test_Y = np.concatenate((label1[(num_train + num_val):, :],
                            label2[(num_train + num_val):, :],
                            label3[(num_train + num_val):, :]))

    return (train_X, train_Y, val_X, val_Y, test_X, test_Y)



if __name__ == '__main__':
    num_sample = 3000
    num_train = 2500
    num_val = 250
    (data1, label1, data2, label2, data3, label3) = create_data(num_sample)
    (train_X, train_Y, val_X, val_Y, test_X, test_Y) = create_train_val_test(data1, label1, data2, label2, data3,
                                                                             label3,
                                                                             num_train, num_val)

    # Pre-process data
    mean_X = np.mean(train_X, 0, keepdims=True)
    std_X = np.std(train_X, 0, keepdims=True)
    train_X = (train_X - mean_X) / std_X
    val_X = (val_X - mean_X) / std_X
    test_X = (test_X - mean_X) / std_X


    config = {}
    config['num_epoch'] = 1000
    config['lr'] = 0.8
    config['num_train_per_class'] = num_train
    config['num_hidden_node'] = 24
    config['display_rate'] = 10 # epochs per display time
    # BasicSGDDemo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

    config['momentum'] = 0.9
    # BasicSGDMomentumDemo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

    config['ada_epsilon'] = np.asarray(0.00000001) # 10^-8
    # BasicAdagradDemo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

    config['adam_beta1'] = np.asarray(0.9)  # 10^-8
    config['adam_beta2'] = np.asarray(0.999)  # 10^-8
    BasicAdamDemo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

    ### Uncomment the below code blocks to view data

    # visualize_data(data1, data2, data3, 1)
    # visualize_data(train_X[0:num_train,:],
    #                train_X[num_train:num_train*2,:],
    #                train_X[num_train*2:,:],
    #                2)
    #
    # visualize_data(val_X[0:num_val, :],
    #                val_X[num_val:num_val * 2, :],
    #                val_X[num_val * 2:, :],
    #                3)
    #
    # visualize_data(test_X[0:num_val, :],
    #                test_X[num_val:num_val * 2, :],
    #                test_X[num_val * 2:, :],
    #                4)
    # pylab.ion()
    # pylab.show()

    train_Y = np
