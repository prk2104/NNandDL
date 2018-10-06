import _pickle as cPickle
import gzip

import numpy as np


def load_data():
    f = gzip.open('/Chapter-1/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()

    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(ti, (784, 1)) for ti in tr_d[0]]
    training_results = [vectorized_result(r) for r in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(vi, (784, 1)) for vi in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(ti, (784, 1)) for ti in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)


def vectorized_result(num):
    result = np.zeros((10, 1))
    result[num] = 1.0
    return result
