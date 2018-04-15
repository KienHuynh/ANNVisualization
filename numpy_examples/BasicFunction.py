import numpy as np

# Basic functions
def sigmoid(X, bp = False):
    if (not bp):
        return 1.0/(1.0 + np.exp(-X))
    else:
        sig = 1.0/(1.0 + np.exp(-X))
        return sig*(1-sig)

def softmax(X):
    # Assume that the second dim is the feature dim
    max_input = np.max(X, 1, keepdims=True)
    X_max = X - max_input
    e = np.exp(X_max)
    sum_e = np.sum(e, 1, keepdims=True)
    return e / sum_e

def relu(X, bp = False):
    result = X
    if (not bp):
        result = X
        result[X < 0] = 0
    else:
        result[X > 0] = 1
        result[X <= 0] = 0
    return result

def activation_function(X, type, bp = False):
    if (type == "sigmoid"):
        return sigmoid(X, bp)
    elif (type == "relu"):
        return relu(X, bp)
    else:
        raise ValueError("Activation function not recognized")

def softmax_log_loss(X, Y, bp=False):
    """
    Calculate softmax log loss (aka categorical cross entropy after softmax)

    :type X: 2D numpy array
    :param X: the predictions/labels computed by the network

    :type Y: 2D numpy array
    :param Y: the groundtruth, what we want X to be
    """
    # Perform checking
    assert len(X.shape) == 2, "X should have a shape of (num_sample, num_class)"
    assert len(Y.shape) == 2, "Y should have a shape of (num_sample, num_class)"
    assert (X.shape[0] == Y.shape[0]) and (X.shape[1] == Y.shape[1]), "Predictions and labels should have the same shape"
    n = Y.shape[0]

    if (not bp):
        # Perform feedforward
        # Assume that the second dim is the feature dim
        xdev = X - np.max(X, 1, keepdims=True)
        lsm = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
        return -np.sum(lsm*Y) / n
    else:
        # Perform backprob and return the derivatives
        xmax = np.max(X, 1, keepdims=True)
        ex = np.exp(X-xmax)
        dFdX = ex/np.sum(ex, 1, keepdims=True)
        dFdX[Y.astype(bool)] = (dFdX[Y.astype(bool)]-1)
        dFdX = dFdX / n
        # dFdX = (dFdX - 1) / n
        return dFdX

