Artificial neural network visualization
=======================================

This is a project dedicated to visualize ANN and its related processes. It is used for studying purpose, not for real applications.

![](img/sgd.gif?raw=true)
![](img/sgdm.gif?raw=true)
![](img/sgdm-kernel.gif?raw=true)

Files:

- ClassifyNNSGD_main.py: Provide basic data for classification using ANN. We can tweak the hyper params and change SGD algorithms to see various learning outcomes.
- ClassifyNNKernel_main.py: Provide basic data to demonstrate the effect of using polynomial kernel on input data. Change the poly order to see how they work out.
- SGD.py: The main file that offers (stochastic) gradient descent algorithms to train an ANN. Currently, only batch gd is implemented.
- BasicFunction.py: Basic function for ANNs.
- Utility.py

You can modify params such as num_hidden_node, lr (learning rate), num_sample, etc. to see their affects

Requirements:
- Python 2.7
- Numpy http://www.scipy.org/scipylib/download.html
- Matplotlib http://matplotlib.org/users/installing.html
