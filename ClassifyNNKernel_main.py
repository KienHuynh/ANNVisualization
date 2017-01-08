import numpy as np
from SGD import *

rng = np.random.RandomState(1311)
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
    m2 = np.asarray([0.1,0.1], dtype=np.float32)
    cov2 = np.asarray([[0.5, 0], [0, 0.5]], dtype=np.float32)
    data2 = rng.multivariate_normal(m2, cov2, num_sample)
    label2 = np.ones((num_sample), dtype=np.uint16)
    label2 = I[label2, :]


    return (data1, label1, data2, label2)

def create_train_val_test(data1, label1, data2, label2, num_train, num_val):
    train_X = np.concatenate((data1[0:num_train, :],
                              data2[0:num_train, :]))

    train_Y = np.concatenate((label1[0:num_train, :],
                              label2[0:num_train, :]))

    val_X = np.concatenate((data1[num_train:(num_train+num_val), :],
                              data2[num_train:(num_train+num_val), :]))

    val_Y = np.concatenate((label1[num_train:(num_train+num_val), :],
                              label2[num_train:(num_train+num_val), :]))

    test_X = np.concatenate((data1[(num_train + num_val):, :],
                            data2[(num_train + num_val):, :]))

    test_Y = np.concatenate((label1[(num_train + num_val):, :],
                            label2[(num_train + num_val):, :]))

    return (train_X, train_Y, val_X, val_Y, test_X, test_Y)

def kernel_preprocess(data1, data2, poly_order):
    n = data1.shape[0]
    for i in range(2,poly_order):
        power1 = i
        power2 = 0
        for j in range(0, i+1):
            data1_new = (data1[:, 0] ** power1).reshape((n, 1))
            data1 = np.concatenate((data1, data1_new), 1)
            data2_new = (data2[:, 0] ** power2).reshape((n, 1))
            data2 = np.concatenate((data2, data2_new), 1)
            power1 -= 1
            power2 += 1

    return (data1, data2)

if __name__ == '__main__':
    num_sample = 10
    num_train = 7
    num_val = 3
    (data1, label1, data2, label2) = create_data(num_sample)
    a = kernel_preprocess(data1, data2, 3)
    (train_X, train_Y, val_X, val_Y, test_X, test_Y) = create_train_val_test(data1, label1, data2, label2,
                                                                             num_train, num_val)



    # Pre-process data
    mean_X = np.mean(train_X, 0, keepdims=True)
    std_X = np.std(train_X, 0, keepdims=True)
    train_X = (train_X - mean_X) / std_X
    val_X = (val_X - mean_X) / std_X
    test_X = (test_X - mean_X) / std_X


    config = {}
    config['demo_type'] = "classifynnkernel"
    config['num_epoch'] = 1000
    config['lr'] = 0.8
    config['num_train_per_class'] = num_train
    config['num_hidden_node'] = 2
    config['display_rate'] = 10 # epochs per display time
    # basic_sgd_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

    config['momentum'] = 0.9
    basic_sgd_momentum_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

    config['ada_epsilon'] = np.asarray(0.00000001) # 10^-8
    # BasicAdagradDemo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

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
