__author__ = 'DY'
# Run some setup code for this notebook. Don't modify anything in this cell.

import random
import numpy as np
#from cs224d.data_utils import *
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

def softmax(x):
    """ Softmax function """
    ###################################################################
    # Compute the softmax function for the input here.                #
    # It is crucial that this function is optimized for speed because #
    # it will be used frequently in later code.                       #
    # You might find numpy functions np.exp, np.sum, np.reshape,      #
    # np.max, and numpy broadcasting useful for this task. (numpy     #
    # broadcasting documentation:                                     #
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)  #
    # You should also make sure that your code works for one          #
    # dimensional inputs (treat the vector as a row), you might find  #
    # it helpful for your later problems.                             #
    ###################################################################

    ### YOUR CODE HERE
    if (len(x.shape) == 1):
        x = lift(x)
    x = norm(x)
    x = np.exp(x)
    #print x
    xsum = np.sum(x, axis=1)
    #print xsum
    #x = (x.T / xsum).T
    x = x / lift(xsum).T

    ### END YOUR CODE

    return x

def norm(x):
    x = x - lift(np.max(x, axis=1)).T
    return x

def lift(x):
    return x.reshape((1,) + x.shape)

def sigmoid(x):
    """ Sigmoid function """
    ###################################################################
    # Compute the sigmoid function for the input here.                #
    ###################################################################

    ### YOUR CODE HERE
    x = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return x

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    ###################################################################
    # Compute the gradient for the sigmoid function here. Note that   #
    # for this implementation, the input f should be the sigmoid      #
    # function value of your original input x.                        #
    ###################################################################

    ### YOUR CODE HERE
    f = f * (1.0 - f)
    ### END YOUR CODE

    return f

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    #print fx
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### YOUR CODE HERE: try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it
        ### possible to test cost functions with built in randomness later

        #return # replace this line with your code

        x[ix] = x[ix] + h;
        a = f(x)[0]
        x[ix] = x[ix] - h;
        x[ix] = x[ix] - h;
        b = f(x)[0]
        x[ix] = x[ix] + h;
        numgrad = (a - b) / 2.0 / h
        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"
# quad = lambda x: (np.sum(x ** 2), x * 2)
# gradcheck_naive(quad, np.random.randn(4,5))
# gradcheck_naive(lambda x: (sigmoid(x), sigmoid_grad(sigmoid(x))), np.array(432.423))


# Set up fake data and parameters for the neural network
N = 20
dimensions = [10, 5, 10]
data = np.random.randn(N, dimensions[0])
labels = np.zeros((N, dimensions[2]))
for i in xrange(N):
    labels[i,random.randint(0,dimensions[2]-1)] = 1

params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )


def forward_backward_prop(data, labels, params):
    """ Forward and backward propagation for a two-layer sigmoidal network """
    ###################################################################
    # Compute the forward propagation and for the cross entropy cost, #
    # and backward propagation for the gradients for all parameters.  #
    ###################################################################

    ### Unpack network parameters (do not modify)
    t = 0
    W1 = np.reshape(params[t:t+dimensions[0]*dimensions[1]], (dimensions[0], dimensions[1]))
    t += dimensions[0]*dimensions[1]
    b1 = np.reshape(params[t:t+dimensions[1]], (1, dimensions[1]))
    t += dimensions[1]
    W2 = np.reshape(params[t:t+dimensions[1]*dimensions[2]], (dimensions[1], dimensions[2]))
    t += dimensions[1]*dimensions[2]
    b2 = np.reshape(params[t:t+dimensions[2]], (1, dimensions[2]))

    ### YOUR CODE HERE: forward propagation
    if (len(data.shape) == 1):
        data = lift(data)
        labels = lift(labels)
    # cost = ...

    N = data.shape[0]

    Z0 = data
    A0 = norm(Z0)
    Z1 = np.dot(A0, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    Z3 = A2
    A3 = CE(Z3, labels)
    cost = np.sum(A3) / N

    grad = np.zeros(params.shape)
    # gradW1 = np.zeros(W1.shape)
    # gradb1 = np.zeros(b1.shape)
    # gradW2 = np.zeros(W2.shape)
    # gradb2 = np.zeros(b2.shape)
    for data_idx in range(N):
        label = labels[data_idx]
        z0 = Z0[data_idx]
        a0 = A0[data_idx]
        z1 = Z1[data_idx]
        a1 = A1[data_idx]
        z2 = Z2[data_idx]
        a2 = A2[data_idx]
        z3 = Z3[data_idx]
        a3 = A3[data_idx]
        ### END YOUR CODE

        ### YOUR CODE HERE: backward propagation

        #layer 3
        R = 1
        C = z3.shape[0]
        gradz3 = -label / z3 # [-labels[i] / z3[i] for i in len(z3)]


        #layer 2
        R = z3.shape[0]
        C = z2.shape[0]

        # gradz3_z2 = [[z3[j] * (1 - z3[k]) if k == j else z3[j] * -z3[k]
        #               for k in range(C)]
        #              for j in range(R)]
        # gradz3_z2 = np.array(gradz3_z2)
        # print gradz3_z2

        grada2 = gradz3;

        R = a2.shape[0]
        C = z2.shape[0]
        grada2_z2 = np.dot(lift(a2).T, -lift(a2)) + np.diag(a2)
        # print grada2_z2

        R = 1
        C = z2.shape[0]
        gradz2 = np.dot(grada2, grada2_z2)
        # print gradz2

        # R = z2.shape[0]
        # C = W2.shape[0]
        # C1 = W2.shape[1]
        # gradz2_W2 = [[[a1[k] if l == j else 0
        #                for l in range(C1)]
        #               for k in range(C)]
        #              for j in range(R)]
        # gradz2_W2 = np.array(gradz2_W2)
        #
        # R = W2.shape[0]
        # C = W2.shape[1]
        # gradW2 = np.tensordot(gradz2, gradz2_W2, axes=1)
        # print gradW2
        gradW2 = np.dot(np.reshape(a1.T, (a1.shape[0], 1)), np.reshape(gradz2, (1, gradz2.shape[0])))
        # print gradW2

        R = 1
        C = b2.shape[0]
        gradb2 = gradz2 # omit * identity

        #layer 1
        R = z2.shape[0]
        C = a1.shape[0]
        gradz2_a1 = W2.T
        # gradz2_a1 = np.array(gradz2_a1)

        R = a1.shape[0]
        C = z1.shape[0]
        # grada1_z1 = [[a1[j] * (1 - a1[k]) if k == j else 0
        #               for k in range(C)]
        #              for j in range(R)]
        # grada1_z1 = np.array(grada1_z1)
        # print grada1_z1

        grada1_z1 = np.diag(a1 * (1 - a1))
        # print grada1_z1

        R = z2.shape[0]
        C = z1.shape[0]
        gradz2_z1 = np.dot(gradz2_a1, grada1_z1)
        gradz2_z1 = np.array(gradz2_z1)

        R = 1
        C = z1.shape[0]
        gradz1 = np.dot(gradz2, gradz2_z1)
        gradz1 = np.array(gradz1)

        # R = z1.shape[0]
        # C = W1.shape[0]
        # C1 = W1.shape[1]
        # gradz1_W1 = [[[a0[k] if l == j else 0
        #                 for l in range(C1)]
        #                for k in range(C)]
        #               for j in range(R)]
        # gradz1_W1 = np.array(gradz1_W1)
        #
        # R = W1.shape[0]
        # C = W1.shape[1]
        # gradW1 = np.tensordot(gradz1,gradz1_W1, axes=1)
        # print gradW1
        gradW1 = np.dot(np.reshape(a0.T, (a0.shape[0], 1)), np.reshape(gradz1, (1, gradz1.shape[0])))
        # print gradW1

        gradb1 = gradz1

        grad += np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    # grad = np.array(grad)
    grad = grad / N
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    #grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def CE(x, y):
    return np.array([np.dot(-y[i], np.log(x[i]))
            for i in range(x.shape[0])])

# Perform gradcheck on your neural network
print "=== For autograder ==="
gradcheck_naive(lambda params: forward_backward_prop(data, labels, params), params)
