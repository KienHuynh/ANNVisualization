import numpy as np
from SGD import *
from Utility import *

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
    m1 = np.asarray([0.5, 0.5], dtype=np.float32)
    cov1 = np.asarray([[0.1, 0],
                       [0, 0.1]], dtype=np.float32)
    data1 = rng.multivariate_normal(m1, cov1, num_sample)
    label1 = np.ones((num_sample), dtype=np.uint16) - 1
    label1 = I[label1,:]

    # Generate second class
    m2 = np.asarray([0.3,0.3], dtype=np.float32)
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



if __name__ == '__main__':
    num_sample = 7
    num_train = 4
    num_val = 3

    config = {}
    config['demo_type'] = "classifynnkernel"
    config['save_img'] = True
    config['num_epoch'] = 10000
    config['kernel_poly_order'] = 7
    (data1, label1, data2, label2) = create_data(num_sample)
    data1 = kernel_preprocess(data1, config['kernel_poly_order'])
    data2 = kernel_preprocess(data2, config['kernel_poly_order'])
    (train_X, train_Y, val_X, val_Y, test_X, test_Y) = create_train_val_test(data1, label1, data2, label2,
                                                                             num_train, num_val)

    # Pre-process data
    mean_X = np.mean(train_X, 0, keepdims=True)
    std_X = np.std(train_X, 0, keepdims=True)
    train_X = (train_X - mean_X) / std_X
    val_X = (val_X - mean_X) / std_X
    test_X = (test_X - mean_X) / std_X

    config['lr'] = 0.1
    config['num_train_per_class'] = num_train
    config['num_hidden_node'] = 1
    config['activation_function'] = 'relu'
    config['display_rate'] = 15 # epochs per display time
    # basic_sgd_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

    config['momentum'] = 0.9
    basic_sgd_momentum_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

    config['ada_epsilon'] = np.asarray(0.00000001) # 10^-8
    # basic_adagrad_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

    config['adam_beta1'] = np.asarray(0.9)  # 10^-8
    config['adam_beta2'] = np.asarray(0.999)  # 10^-8
    # basic_adam_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config)

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
