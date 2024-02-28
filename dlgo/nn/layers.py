import numpy as np
# end::imports[]


# tag::sigmoid[]
def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)
# end::sigmoid[]


# tag::sigmoid_prime[]
def sigmoid_prime_double(x):
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)
# end::sigmoid_prime[]

a = sigmoid(5)