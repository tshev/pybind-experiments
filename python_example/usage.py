import python_example
import numpy as np
import timeit
n = 10**7
a = np.random.random(n)
b = np.random.random(n)

def dot_product(x, y):
    r = 0.0
    for a, b in zip(x, y):
        r += a * b
    return r

print(timeit.timeit(lambda: python_example.dot_product(a, b), number=1))
print(timeit.timeit(lambda: a.dot(b), number=1))
print(timeit.timeit(lambda: dot_product(a, b), number=1))
