import numpy as np
from dnn_utils import *
import h5py
import matplotlib.pyplot as plt

def initialize_parameters_deep(layers_dims, default_constant=0.01, smart_initialize=None):
    """
    :param layers_dims:  a list of dimension of the layers
    :param default_constant: decrease initialed weight to make gradient descend easier
    :param smart_initialize: use smart initialization - kamier for relu, xavier for sigmoid
    :return: parameters that holds weights and biases
    """
    parameters = {}
    L = len(layers_dims)
    if smart_initialize is None:
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * default_constant
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))


    elif smart_initialize == "xavier":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
                6 / (layers_dims[l - 1] + layers_dims[l]))
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    elif smart_initialize == "kaiming":
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / (layers_dims[l - 1]))
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    else:
        print("An error is occured. Please check the parameters")

    assert (parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
    assert (parameters['b' + str(l)].shape == (layers_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    """
    :param A: activations from previous layers
    :param W: weights of the corresponding neuron
    :param b: biases of the corresponding neuron
    :return: computed activations for forward propagation and the cache that we will use in backward propagation
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    assert(Z.shape == (W.shape[0], A.shape[1]))
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation="relu"):
    """
    :param A_prev: activations from the previous neuron
    :param W:  weights of the corresponding connections
    :param b:  biases
    :param activation: activation function that we will use: relu,sigmoid
    :return: computed activations for forward propogation and the cache that we will use in backward propagation
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    else:
        print("An error has occured. Please check the parameters again.")
        return None

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X,parameters):
    """
    :param X: data,numpy array of the shape [features,number_of_examples]
    :param parameters: outputs of initialization
    :return: activation and cache
    """
    caches = []
    L = len(parameters) // 2
    A = X #First activation is the input itself
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters["W" + str(l)],parameters["b" + str(l)],activation="relu")
        caches.append(cache)

    AL,cache = linear_activation_forward(A,parameters["W" + str(L)],parameters["b" + str(L)],activation="sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL,caches

def compute_cost(AL,Y, cost_function = "maximum_likelihood"):
    """
    :param AL: Output of the NN
    :param Y: True label
    :return: the cost computed by cost function. Default is logistic lost
    """
    m = Y.shape[1] # Number of training example
    if cost_function == "maximum_likelihood":
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    return cost



def linear_backward(dZ, cache):
    """
    Backward propogation of single layer
    :param dZ: Gradient of Z with respect to the cost function
    :param cache: tuple of values (A_prev,W,b) that we will catch from backward_propogation
    :return: dA_prev,dW,db
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    :param dA: post-activation gradient for layer l
    :param cache: cache to hold values (linear_cache,activation_cache)
    :param activation: type of activation function  : relu,sigmoid
    :return: dA_prev,dW,db
    """

    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)

    return dA_prev,dW,db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # Compute the dAL

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      'sigmoid')

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

def L_layer_model(X,Y,layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    :param X: Input image
    :param Y:  Output - label
    :param layers_dims:  list containing the dimensions
    :param learning_rate: learning rate of the gradient descent
    :param num_iterations: number of iterations - passes
    :param print_cost: if true, print costs every 100 step
    :return: parameters learnt by model
    """
    costs = []
    parameters = initialize_parameters_deep(layers_dims,smart_initialize="kaiming")

    for i in range(0,num_iterations):
        #Forward propaget
        AL,caches = L_model_forward(X,parameters)

        #Compute cost
        cost = compute_cost(AL,Y)
        #Compute gradients
        grads = L_model_backward(AL,Y,caches)

        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost and i % 100 == 0:
            costs.append(cost)
            print("Iteration no : ",i," cost : ",cost)

    #plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteartions')
    plt.show()
    return parameters

def predict(X,y,parameters):
    """
    :param X: x
    :param y:  y işte başka ne olacak amına koyayım
    :param parameters: bulduk lan amk
    :return:
    """
    L = len(parameters) // 2
    preds,cache = L_model_forward(X,parameters)
    m = preds.shape[1] #number of training example
    p = np.zeros((1,m))
    for i in range(m): #should be vectorized since we are using a fucking loop.
        if preds[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("Accuracy of the model is : ", np.sum(p==y,axis=1)/m)
    return p


