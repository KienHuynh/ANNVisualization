import numpy as np
import matplotlib.pyplot as plt
import pylab
from BasicFunction import *
from Utility import *
import os

rng = np.random.RandomState(1311)
def visualize_data(data1, data2, data3, figure_num):

    if (data1.shape[0] > 1000):
        point_size = 10
    else:
        point_size = 50

    plt.figure(figure_num)
    plt.axis('equal')
    plt.scatter(data1[:,0], data1[:,1], point_size, 'r')
    plt.scatter(data2[:, 0], data2[:, 1], point_size, 'g')
    plt.scatter(data3[:, 0], data3[:, 1], point_size, 'b')
    bp = 1

def numerical_grad_check(X, Y, W1, b1, W2, b2, grad, config):
    delta = 0.000005
    activation_function_type = config['activation_function']
    # Check dJ_da2
    # z1 = sigmoid(np.dot(X, W1) + b1)
    # a2 = np.dot(z1, W2) + b2
    #
    # numerical_grad = np.zeros_like(a2)
    # for i in range(0, a2.shape[0]):
    #     for j in range(0, a2.shape[1]):
    #         a2_p = np.copy(a2)
    #         a2_m = np.copy(a2)
    #         a2_p[i,j] = a2_p[i,j] + delta
    #         a2_m[i,j] = a2_m[i,j] - delta
    #         J_p = softmax_log_loss(a2_p, Y)
    #         J_m = softmax_log_loss(a2_m, Y)
    #         numerical_grad[i,j] = (J_p - J_m)/(2.0*delta)

    # Check dJ_dw2

    # numerical_grad = np.zeros_like(W2)
    # for i in range(0, W2.shape[0]):
    #     for j in range(0, W2.shape[1]):
    #         W2_p = np.copy(W2)
    #         W2_m = np.copy(W2)
    #         W2_p[i, j] = W2_p[i, j] + delta
    #         W2_m[i, j] = W2_m[i, j] - delta
    #
    #         z1 = sigmoid(np.dot(X, W1) + b1)
    #
    #         a2 = np.dot(z1, W2_p) + b2
    #         J_p = softmax_log_loss(a2, Y)
    #
    #         a2 = np.dot(z1, W2_m) + b2
    #         J_m = softmax_log_loss(a2, Y)
    #
    #         numerical_grad[i, j] = (J_p - J_m) / (2.0 * delta)

    # Check dJ_db2

    # numerical_grad = np.zeros_like(b2)
    # for i in range(0, b2.shape[0]):
    #     for j in range(0, W2.shape[1]):
    #         b2_p = np.copy(b2)
    #         b2_m = np.copy(b2)
    #         b2_p[i, j] = b2_p[i, j] + delta
    #         b2_m[i, j] = b2_m[i, j] - delta
    #
    #         z1 = sigmoid(np.dot(X, W1) + b1)
    #
    #         a2 = np.dot(z1, W2) + b2_p
    #         J_p = softmax_log_loss(a2, Y)
    #
    #         a2 = np.dot(z1, W2) + b2_m
    #         J_m = softmax_log_loss(a2, Y)
    #
    #         numerical_grad[i, j] = (J_p - J_m) / (2.0 * delta)

    # Check dJ_dz1

    # z1 = sigmoid(np.dot(X, W1) + b1)
    # numerical_grad = np.zeros_like(z1)
    # for i in range(0, z1.shape[0]):
    #     for j in range(0, z1.shape[1]):
    #         z1_p = np.copy(z1)
    #         z1_m = np.copy(z1)
    #         z1_p[i,j] = z1_p[i,j] + delta
    #         z1_m[i,j] = z1_m[i,j] - delta
    #         a2_p = np.dot(z1_p, W2) + b2
    #         a2_m = np.dot(z1_m, W2) + b2
    #         J_p = softmax_log_loss(a2_p, Y)
    #         J_m = softmax_log_loss(a2_m, Y)
    #         numerical_grad[i,j] = (J_p - J_m)/(2.0*delta)


    # Check dJ_da1
    # a1 = np.dot(X, W1) + b1
    # numerical_grad = np.zeros_like(a1)
    # for i in range(0, a1.shape[0]):
    #     for j in range(0, a1.shape[1]):
    #         a1_p = np.copy(a1)
    #         a1_m = np.copy(a1)
    #         a1_p[i, j] = a1_p[i, j] + delta
    #         a1_m[i, j] = a1_m[i, j] - delta
    #
    #         z1 = sigmoid(a1_p)
    #         a2 = np.dot(z1, W2) + b2
    #         J_p = softmax_log_loss(a2, Y)
    #
    #         z1 = sigmoid(a1_m)
    #         a2 = np.dot(z1, W2) + b2
    #         J_m = softmax_log_loss(a2, Y)
    #
    #         numerical_grad[i, j] = (J_p - J_m) / (2.0 * delta)

    # Check dJ_dW1
    # numerical_grad = np.zeros_like(W1)
    # for i in range(0, W1.shape[0]):
    #     for j in range(0, W1.shape[1]):
    #         W1_p = np.copy(W1)
    #         W1_m = np.copy(W1)
    #         W1_p[i, j] = W1_p[i, j] + delta
    #         W1_m[i, j] = W1_m[i, j] - delta
    #
    #         z1 = sigmoid(np.dot(X, W1_p) + b1)
    #         a2 = np.dot(z1, W2) + b2
    #         J_p = softmax_log_loss(a2, Y)
    #
    #         z1 = sigmoid(np.dot(X, W1_m) + b1)
    #         a2 = np.dot(z1, W2) + b2
    #         J_m = softmax_log_loss(a2, Y)
    #
    #         numerical_grad[i, j] = (J_p - J_m) / (2.0 * delta)

    # Check dJ_db1
    numerical_grad = np.zeros_like(b1)
    for i in range(0, b1.shape[0]):
        for j in range(0, b1.shape[1]):
            b1_p = np.copy(b1)
            b1_m = np.copy(b1)
            b1_p[i, j] = b1_p[i, j] + delta
            b1_m[i, j] = b1_m[i, j] - delta

            z1 = activation_function(np.dot(X, W1) + b1_p, activation_function_type)
            a2 = np.dot(z1, W2) + b2
            J_p = softmax_log_loss(a2, Y)

            z1 = activation_function(np.dot(X, W1) + b1_m, activation_function_type)
            a2 = np.dot(z1, W2) + b2
            J_m = softmax_log_loss(a2, Y)

            numerical_grad[i, j] = (J_p - J_m) / (2.0 * delta)

    grad_diff = np.sum(np.abs((numerical_grad - grad)))
    print ("Difference in gradient: %f \n" % (grad_diff))


def find_decision_boundary(X, Y, W1, b1, W2, b2, config):
    num_fake_data = 150
    num_train_sample = X.shape[0]
    num_feature = X.shape[1]
    num_hidden_node = W1.shape[1]
    num_class = Y.shape[1]

    demo_type = config['demo_type']
    activation_function_type = config['activation_function']
    min_X1 = np.min(X[:, 0]) - 0.1
    max_X1 = np.max(X[:, 0]) + 0.1
    min_X2 = np.min(X[:, 1]) - 0.1
    max_X2 = np.max(X[:, 1]) + 0.1

    # Create grid data
    X1 = np.linspace(min_X1, max_X1, num_fake_data)
    X2 = np.linspace(min_X2, max_X2, num_fake_data)
    xv, yv = np.meshgrid(X1, X2)
    xv = xv.flatten().astype(np.float32).reshape((num_fake_data ** 2, 1))
    yv = yv.flatten().astype(np.float32).reshape((num_fake_data ** 2, 1))
    X = np.concatenate((xv, yv), 1)
    if (demo_type == "classifynnkernel"):
        kernel_poly_order = config['kernel_poly_order']
        X = kernel_preprocess(X, kernel_poly_order)

    del xv
    del yv

    a1 = np.dot(X, W1) + b1
    z1 = activation_function(a1, activation_function_type)
    a2 = np.dot(z1, W2) + b2
    # pred = np.repeat(np.argmax(a2, 1).reshape((num_fake_data**2, 1)), 2, 1)
    pred = np.argmax(a2, 1)
    grid0 = X[pred == 0, :]
    grid1 = X[pred == 1, :]
    grid2 = X[pred == 2, :]

    grid0 = grid0.reshape((grid0.shape[0], num_feature))
    grid1 = grid1.reshape((grid1.shape[0], num_feature))
    grid2 = grid2.reshape((grid2.shape[0], num_feature))

    return (grid0, grid1, grid2)


def visualize_decision_grid(data1, data2, data3, figure_num):
    plt.figure(figure_num)
    plt.axis('equal')
    plt.scatter(data1[:, 0], data1[:, 1], 10, c=[1, 0.5, 0.5], marker='+')
    plt.scatter(data2[:, 0], data2[:, 1], 10, c=[0.5, 1, 0.5], marker='+')
    plt.scatter(data3[:, 0], data3[:, 1], 10, c=[0.5, 0.5, 1], marker='+')
    bp = 1


def get_grad(X, Y, W1, b1, W2, b2, config):
    activation_function_type = config['activation_function']

    num_train_sample = X.shape[0]
    num_feature = X.shape[1]
    num_hidden_node = W1.shape[1]
    num_class = Y.shape[1]

    a1 = np.dot(X, W1) + b1
    z1 = activation_function(a1, activation_function_type)
    a2 = np.dot(z1, W2) + b2

    # Calculate W2 and b2 gradient
    dJ_da2 = softmax_log_loss(a2, Y, True)
    # These are basically dJ_da2 but are repeated so we can multiply them with dJ_dW2 and dJ_dz1
    dJ_da2b = np.sum(dJ_da2, 0, keepdims=True)
    dJ_da2W = np.repeat(dJ_da2.reshape((num_train_sample, 1, num_class)), num_hidden_node, 1)
    dJ_da2z1 = np.repeat(dJ_da2.reshape((num_train_sample, 1, num_class)), num_hidden_node, 1)

    da2_dW2 = np.repeat(z1.reshape((num_train_sample, num_hidden_node, 1)), num_class, 2)
    da2_db2 = 1
    da2_dz1 = np.repeat(W2.reshape(1, num_hidden_node, num_class), num_train_sample, 0)

    dJ_dW2 = np.sum(dJ_da2W * da2_dW2, 0)
    dJ_db2 = da2_db2 * dJ_da2b
    dJ_dz1 = np.sum(dJ_da2z1 * da2_dz1, 2)

    # Calculate W1 and b1 gradient
    dJ_dz1_dW1 = np.repeat(dJ_dz1.reshape((num_train_sample, 1, num_hidden_node)), num_feature, 1)
    dz1_da1 = activation_function(a1, activation_function_type, True)
    dz1_da1_W1 = np.repeat(dz1_da1.reshape((num_train_sample, 1, num_hidden_node)), num_feature, 1)
    da1_dW1 = np.repeat(X.reshape((num_train_sample, num_feature, 1)), num_hidden_node, 2)
    da1_db1 = 1

    dJ_dW1 = np.sum(dJ_dz1_dW1 * dz1_da1_W1 * da1_dW1, 0)
    dJ_db1 = np.sum(dJ_dz1 * dz1_da1 * da1_db1, 0, keepdims=True)

    # NumericalGradientCheck(X, Y, W1, b1, W2, b2, dJ_db1)

    return (dJ_dW1, dJ_db1, dJ_dW2, dJ_db2)

def train_draw(X, Y, W1, b1, W2, b2, config, all_cost, i, J):
    # Display the training process

    num_hidden_node = config['num_hidden_node']
    num_train_per_class = config['num_train_per_class']
    train_method = config['train_method']
    save_img = config['save_img']
    demo_type = config['demo_type']

    lr = config['lr']

    pylab.clf()
    f = plt.figure(2, figsize=(16, 8))

    title = '%s with %d hidden nodes, lr = %.4g, %d epoch, cost = %.4g' % (train_method, num_hidden_node, lr, i, J)

    if (train_method == 'sgdm'):
        momentum_rate = config['momentum']
        title += '\nmomentum rate = %.4g' % momentum_rate
        # f.suptitle(title, fontsize=15)

    elif (train_method == 'ada'):
        epsilon = config['ada_epsilon']
        title += '\nepsilon = %.4g' % epsilon

    elif (train_method == 'adam'):
        beta1 = config['adam_beta1']
        beta2 = config['adam_beta2']
        epsilon = config['ada_epsilon']
        title += '\nepsilon = %.4g, beta1 = %.4g, beta2 = %.4g' % (epsilon, beta1, beta2)

    plt.subplot(1, 2, 1)
    [grid1, grid2, grid3] = find_decision_boundary(X, Y, W1, b1, W2, b2, config)

    if (demo_type == "classifynnkernel"):
        grid1 = grid1[:, 0:2]
        grid2 = grid2[:, 0:2]
        grid3 = grid3[:, 0:2]
        X = X[:,0:2]
        title += ' kernel_poly_order = %d' % config['kernel_poly_order']

    visualize_decision_grid(grid1, grid2, grid3, 2)

    visualize_data(X[0:num_train_per_class, :],
                   X[num_train_per_class:num_train_per_class * 2, :],
                   X[num_train_per_class * 2:, :],
                   2)
    plt.subplot(1, 2, 2)
    plt.plot(all_cost, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')

    f.suptitle(title, fontsize=15)

    if (save_img):
        save_img_path = 'giffolder/%s/'%train_method
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        f.savefig(save_img_path + ('/%04d.png' % i), bbox_inches='tight')

    pylab.draw()

def basic_sgd_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config):
    # This is to demonstrate the process of learning a simple one hidden-layer NN
    # Input kernel: linear
    # Num hidden layer: 1
    # Learning method: SGD

    # Parse param from config
    lr = config['lr']
    num_epoch = config['num_epoch']
    num_train_per_class = config['num_train_per_class']
    num_hidden_node = config['num_hidden_node']
    display_rate = config['display_rate']
    activation_function_type = config['activation_function']

    num_train_sample = train_X.shape[0]
    num_feature = train_X.shape[1]
    num_class = train_Y.shape[1]

    # Create a weight matrix of shape (2, num_hidden_node)
    W1 = rng.randn(num_feature, num_hidden_node)
    b1 = rng.randn(1, num_hidden_node)

    # Create output weight
    W2 = rng.randn(num_hidden_node, num_class)
    b2 = rng.randn(1, num_class)

    num_train_sample = 1
    pylab.ion()
    pylab.show()
    all_cost = []
    for i in range(0, num_epoch):
        # Calculate the loss
        a1 = np.dot(train_X, W1) + b1
        z1 = activation_function(a1, activation_function_type)
        a2 = np.dot(z1, W2) + b2
        J = softmax_log_loss(a2, train_Y)

        # Doing backprop
        print('[Epoch %d] Train loss: %f' % (i, J))

        dJ_dW1, dJ_db1, dJ_dW2, dJ_db2 = get_grad(train_X, train_Y, W1, b1, W2, b2, config)
        # NumericalGradientCheck(train_X, train_Y, W1, b1, W2, b2, dJ_db1)
        W1 = W1 - dJ_dW1 * lr
        b1 = b1 - dJ_db1 * lr
        W2 = W2 - dJ_dW2 * lr
        b2 = b2 - dJ_db2 * lr

        all_cost.append(J)

        if (i % display_rate == 0):
            config['train_method'] = 'sgd'
            train_draw(train_X, train_Y, W1, b1, W2, b2, config, all_cost, i, J)

        bp = 1


def basic_sgd_momentum_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config):
    # This is to demonstrate the process of learning a simple one hidden-layer NN
    # Input kernel: linear
    # Num hidden layer: 1
    # Learning method: SGD with momentum

    # Parse param from config
    lr = config['lr']
    num_epoch = config['num_epoch']
    num_train_per_class = config['num_train_per_class']
    num_hidden_node = config['num_hidden_node']
    momentum_rate = config['momentum']
    display_rate = config['display_rate']
    activation_function_type = config['activation_function']

    num_train_sample = train_X.shape[0]
    num_feature = train_X.shape[1]
    num_class = train_Y.shape[1]

    # Create a weight matrix of shape (2, num_hidden_node)
    W1 = rng.randn(num_feature, num_hidden_node)
    b1 = rng.randn(1, num_hidden_node)

    # Create output weight
    W2 = rng.randn(num_hidden_node, num_class)
    b2 = rng.randn(1, num_class)

    # Create momentum storage
    W1m = np.zeros_like(W1)
    b1m = np.zeros_like(b1)
    W2m = np.zeros_like(W2)
    b2m = np.zeros_like(b2)

    num_train_sample = 1
    pylab.ion()
    pylab.show()
    all_cost = []
    for i in range(0, num_epoch):
        # Calculate the loss
        a1 = np.dot(train_X, W1) + b1
        z1 = activation_function(a1, activation_function_type)
        a2 = np.dot(z1, W2) + b2
        J = softmax_log_loss(a2, train_Y)

        # Doing backprop
        print('[Epoch %d] Train loss: %f' % (i, J))

        dJ_dW1, dJ_db1, dJ_dW2, dJ_db2 = get_grad(train_X, train_Y, W1, b1, W2, b2, config)
        # NumericalGradientCheck(train_X, train_Y, W1, b1, W2, b2, dJ_db1)

        W1m = W1m * momentum_rate + lr * dJ_dW1 * lr
        b1m = b1m * momentum_rate + lr * dJ_db1 * lr
        W2m = W2m * momentum_rate + lr * dJ_dW2 * lr
        b2m = b2m * momentum_rate + lr * dJ_db2 * lr

        W1 = W1 - W1m
        b1 = b1 - b1m
        W2 = W2 - W2m
        b2 = b2 - b2m

        all_cost.append(J)

        if (i % display_rate == 0):
            config['train_method'] = 'sgdm'
            train_draw(train_X, train_Y, W1, b1, W2, b2, config, all_cost, i, J)

        bp = 1


def basic_adagrad_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config):
    # This is to demonstrate the process of learning a simple one hidden-layer NN
    # Input kernel: linear
    # Num hidden layer: 1
    # Learning method: Adagrad

    # Parse param from config
    lr = config['lr']
    num_epoch = config['num_epoch']
    num_train_per_class = config['num_train_per_class']
    num_hidden_node = config['num_hidden_node']
    epsilon = config['ada_epsilon']
    display_rate = config['display_rate']
    activation_function_type = config['activation_function']

    num_train_sample = train_X.shape[0]
    num_feature = train_X.shape[1]
    num_class = train_Y.shape[1]

    # Create a weight matrix of shape (2, num_hidden_node)
    W1 = rng.randn(num_feature, num_hidden_node)
    b1 = rng.randn(1, num_hidden_node)

    # Create output weight
    W2 = rng.randn(num_hidden_node, num_class)
    b2 = rng.randn(1, num_class)

    # Create accumulative gradient storage
    W1g = np.zeros_like(W1)
    b1g = np.zeros_like(b1)
    W2g = np.zeros_like(W2)
    b2g = np.zeros_like(b2)

    num_train_sample = 1
    pylab.ion()
    pylab.show()
    all_cost = []
    for i in range(0, num_epoch):
        # Calculate the loss
        a1 = np.dot(train_X, W1) + b1
        z1 = activation_function(a1, activation_function_type)
        a2 = np.dot(z1, W2) + b2
        J = softmax_log_loss(a2, train_Y)

        # Doing backprop
        print('[Epoch %d] Train loss: %f' % (i, J))

        dJ_dW1, dJ_db1, dJ_dW2, dJ_db2 = get_grad(train_X, train_Y, W1, b1, W2, b2, config)
        # NumericalGradientCheck(train_X, train_Y, W1, b1, W2, b2, dJ_db1)

        W1g = W1g + dJ_dW1 ** 2
        b1g = b1g + dJ_db1 ** 2
        W2g = W2g + dJ_dW2 ** 2
        b2g = b2g + dJ_db2 ** 2

        W1 = W1 - dJ_dW1 * lr / np.sqrt(W1g + epsilon)
        b1 = b1 - dJ_db1 * lr / np.sqrt(b1g + epsilon)
        W2 = W2 - dJ_dW2 * lr / np.sqrt(W2g + epsilon)
        b2 = b2 - dJ_db2 * lr / np.sqrt(b2g + epsilon)

        all_cost.append(J)

        if (i % display_rate == 0):
            config['train_method'] = 'ada'
            train_draw(train_X, train_Y, W1, b1, W2, b2, config, all_cost, i, J)

        bp = 1

def basic_adam_demo(train_X, train_Y, val_X, val_Y, test_X, test_Y, config):
    # This is to demonstrate the process of learning a simple one hidden-layer NN
    # Input kernel: linear
    # Num hidden layer: 1
    # Learning method: Adagrad

    # Parse param from config
    lr = config['lr']
    num_epoch = config['num_epoch']
    num_train_per_class = config['num_train_per_class']
    num_hidden_node = config['num_hidden_node']
    beta1 = config['adam_beta1']
    beta2 = config['adam_beta2']
    epsilon = config['ada_epsilon']
    display_rate = config['display_rate']
    activation_function_type = config['activation_function']

    num_train_sample = train_X.shape[0]
    num_feature = train_X.shape[1]
    num_class = train_Y.shape[1]

    # Create a weight matrix of shape (2, num_hidden_node)
    W1 = rng.randn(num_feature, num_hidden_node)
    b1 = rng.randn(1, num_hidden_node)

    # Create output weight
    W2 = rng.randn(num_hidden_node, num_class)
    b2 = rng.randn(1, num_class)

    # Create accumulative gradient storage
    W1m = np.zeros_like(W1)
    b1m = np.zeros_like(b1)
    W2m = np.zeros_like(W2)
    b2m = np.zeros_like(b2)

    W1v = np.zeros_like(W1)
    b1v = np.zeros_like(b1)
    W2v = np.zeros_like(W2)
    b2v = np.zeros_like(b2)

    num_train_sample = 1
    pylab.ion()
    pylab.show()
    all_cost = []
    for i in range(0, num_epoch):
        # Calculate the loss
        a1 = np.dot(train_X, W1) + b1
        z1 = activation_function(a1, activation_function_type)
        a2 = np.dot(z1, W2) + b2
        J = softmax_log_loss(a2, train_Y)

        # Doing backprop
        print('[Epoch %d] Train loss: %f' % (i, J))

        dJ_dW1, dJ_db1, dJ_dW2, dJ_db2 = get_grad(train_X, train_Y, W1, b1, W2, b2, config)
        # NumericalGradientCheck(train_X, train_Y, W1, b1, W2, b2, dJ_db1)

        W1m = beta1 * W1m + (1 - beta1) * dJ_dW1
        b1m = beta1 * b1m + (1 - beta1) * dJ_db1
        W2m = beta1 * W2m + (1 - beta1) * dJ_dW2
        b2m = beta1 * b2m + (1 - beta1) * dJ_db2

        W1v = beta2 * W1v + (1 - beta2) * (dJ_dW1**2)
        b1v = beta2 * b1v + (1 - beta2) * (dJ_db1**2)
        W2v = beta2 * W2v + (1 - beta2) * (dJ_dW2**2)
        b2v = beta2 * b2v + (1 - beta2) * (dJ_db2**2)

        W1m_hat = W1m / (1 - beta1 ** (i + 1))
        b1m_hat = b1m / (1 - beta1 ** (i + 1))
        W2m_hat = W2m / (1 - beta1 ** (i + 1))
        b2m_hat = b2m / (1 - beta1 ** (i + 1))

        W1v_hat = W1v / (1 - beta2 ** (i + 1))
        b1v_hat = b1v / (1 - beta2 ** (i + 1))
        W2v_hat = W2v / (1 - beta2 ** (i + 1))
        b2v_hat = b2v / (1 - beta2 ** (i + 1))

        W1 = W1 - lr * W1m_hat / (np.sqrt(W1v_hat) + epsilon)
        b1 = b1 - lr * b1m_hat / (np.sqrt(b1v_hat) + epsilon)
        W2 = W2 - lr * W2m_hat / (np.sqrt(W2v_hat) + epsilon)
        b2 = b2 - lr * b2m_hat / (np.sqrt(b2v_hat) + epsilon)


        all_cost.append(J)

        if (i % display_rate == 0):
            config['train_method'] = 'adam'
            train_draw(train_X, train_Y, W1, b1, W2, b2, config, all_cost, i, J)


        bp = 1
