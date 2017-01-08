import numpy as np

def kernel_preprocess(data, poly_order):
    n = data.shape[0]
    for i in range(2,poly_order):
        power1 = i
        power2 = 0
        for j in range(0, i+1):
            data_new = ((data[:, 0] ** power1) * data[:,1] ** power2).reshape((n, 1))
            data = np.concatenate((data, data_new), 1)
            power1 -= 1
            power2 += 1

    return data
