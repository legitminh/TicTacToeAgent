import numpy as np
import math
def softmax(z):
    z = z - np.max(z)
    exp = np.exp(z)
    return exp / np.sum(exp)

def rd(num, decimal_places=0):
    p = pow(10, decimal_places)
    num *= p
    if math.ceil(num) - num < num - math.floor(num):
        return math.ceil(num) / p
    return math.floor(num) / p

def rd_array(array, decimal_places=0):
    return np.array([rd(x, decimal_places) for x in array])