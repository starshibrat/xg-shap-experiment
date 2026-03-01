import numpy as np

def sigmoid(y):
    return 1/(1 + np.exp(-y))

def largest_divisor(n: int, end_value: int) -> int:
    assert end_value < n
    a = 1
    for i in range(end_value, 1, -1):
        if n % i == 0:
            a = i
            return a
    
    return a