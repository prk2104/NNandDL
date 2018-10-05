"""
This code mainly emphsizes on implementation details for the neural networks.
Vectorization is not applied for fulfilling the same purpose.
"""


import random
import numpy as np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for i in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1}/{2}".format(i,
                                                  self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete.".format(i))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b - (eta/len(nabla_b))*nb for b,
                       nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta.len(nabla_w)*nw for w,
                             nw in zip(self.weights, nabla_w))]


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))
