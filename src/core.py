import numpy as np

def factorisation_machine(X, v, w0, w1 ):
    dot = np.dot
    sum = np.sum
    mul = np.multiply
    interactions = 0.5 * sum((dot(X, np.transpose(v)) ** 2) \
                                   - dot(mul(X, X), np.transpose(v ** 2)), axis=1)
    y_hat = w0[0] + dot(X, w1) + interactions
    
    return y_hat