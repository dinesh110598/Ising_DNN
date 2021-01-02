import math
import numpy as np
from numba import cuda

@cuda.jit
def update_black (spin, seed, T, J):
    m = T.shape[1]
    z, x, y = cuda.grid(3)
    n = int(math.floor (z / m))
    l = z % m
    p, q = x % 2, y % 2

    def random_uniform():
        seed[z, x, y] = np.int32((seed[z ,x, y]*1664525 + 1013904223) % 2**31)
        return seed[z, x, y] / (2**31)

    def bvc (x):
        if x == spin.shape[1]:
            x = 0
        return x       

    def sum_nn():
        value = 0
        value += spin[z, x, bvc(y+1)]
        value += spin[z, x, y-1]
        value += spin[z, bvc(x+1), y]
        value += spin[z, x-1, y]
        return value

    def calc():
        probs = random_uniform()
        if (probs < math.exp(2*(J[0]*sum_nn())*spin[z, x, y]/T[n,l])):
            spin[z, x, y] *= np.int8(-1)

    if (p == 0 and q == 0) or (p == 1 and q == 1):
        calc()

@cuda.jit
def update_white (spin, seed, T, J):
    m = T.shape[1]
    z, x, y = cuda.grid(3)
    n = int(math.floor (z / m))
    l = z % m
    p, q = x % 2, y % 2

    def random_uniform():
        seed[z, x, y] = np.int32((seed[z ,x, y]*1664525 + 1013904223) % 2**31)
        return seed[z, x, y] / (2**31)

    def bvc (x):
        if x == spin.shape[1]:
            x = 0
        return x       

    def sum_nn():
        value = 0
        value += spin[z, x, bvc(y+1)]
        value += spin[z, x, y-1]
        value += spin[z, bvc(x+1), y]
        value += spin[z, x-1, y]
        return value

    def calc():
        probs = random_uniform()
        if (probs < math.exp(2*(J[0]*sum_nn())*spin[z, x, y]/T[n,l])):
            spin[z, x, y] *= np.int8(-1)

    if (p == 0 and q == 1) or (p == 1 and q == 0):
        calc()

@cuda.jit
def update_red (spin, seed, T, J):
    m = T.shape[1]
    z, x, y = cuda.grid(3)
    n = int(math.floor (z / m))
    l = z % m
    p, q = x % 3, y % 2

    def random_uniform():
        seed[z, x, y] = np.int32((seed[z ,x, y]*1664525 + 1013904223) % 2**31)
        return seed[z, x, y] / (2**31)

    def bvc (x):
        if x == spin.shape[1]:
            x = 0
        return x       

    def sum_nn():  # This adds spins of six neighbours instead of 4 subject to
        #many constraints characteristic of triangular lattices
        value = 0.
        if (y % 2 == 0):
            value += spin[z, x, bvc(y+1)]
            value += spin[z, x-1, bvc(y+1)]
            value += spin[z, x, y-1]
            value += spin[z, x-1, y-1]
        else:
            value += spin[z, bvc(x+1), bvc(y+1)]
            value += spin[z, x, bvc(y+1)]
            value += spin[z, bvc(x+1), y-1]
            value += spin[z, x, y-1]

        value += spin[z, bvc(x+1), y]
        value += spin[z, x-1, y]
        return value

    def calc():
        probs = random_uniform()
        if (probs < math.exp(2*J[0]*spin[z, x, y]*sum_nn()/T[n,l])):
            spin[z, x, y] *= np.int8(-1)

    if (p == 0 and q == 0) or (p == 1 and q == 1):
        calc()

@cuda.jit
def update_blue (spin, seed, T, J):
    m = T.shape[1]
    z, x, y = cuda.grid(3)
    n = int(math.floor (z / m))
    l = z % m
    p, q = x % 3, y % 2

    def random_uniform():
        seed[z, x, y] = np.int32((seed[z ,x, y]*1664525 + 1013904223) % 2**31)
        return seed[z, x, y] / (2**31)

    def bvc (x):
        if x == spin.shape[1]:
            x = 0
        return x       

    def sum_nn():  # This adds spins of six neighbours instead of 4 subject to
        #many constraints characteristic of triangular lattices
        value = 0.
        if (y % 2 == 0):
            value += spin[z, x, bvc(y+1)]
            value += spin[z, x-1, bvc(y+1)]
            value += spin[z, x, y-1]
            value += spin[z, x-1, y-1]
        else:
            value += spin[z, bvc(x+1), bvc(y+1)]
            value += spin[z, x, bvc(y+1)]
            value += spin[z, bvc(x+1), y-1]
            value += spin[z, x, y-1]

        value += spin[z, bvc(x+1), y]
        value += spin[z, x-1, y]
        return value

    def calc():
        probs = random_uniform()
        if (probs < math.exp(2*J[0]*spin[z, x, y]*sum_nn()/T[n,l])):
            spin[z, x, y] *= np.int8(-1)

    if (p == 1 and q == 0) or (p == 2 and q == 1):
        calc()

@cuda.jit
def update_green (spin, seed, T, J):
    m = T.shape[1]
    z, x, y = cuda.grid(3)
    n = int(math.floor (z / m))
    l = z % m
    p, q = x % 3, y % 2

    def random_uniform():
        seed[z, x, y] = np.int32((seed[z ,x, y]*1664525 + 1013904223) % 2**31)
        return seed[z, x, y] / (2**31)

    def bvc (x):
        if x == spin.shape[1]:
            x = 0
        return x       

    def sum_nn():  # This adds spins of six neighbours instead of 4 subject to
        #many constraints characteristic of triangular lattices
        value = 0.
        if (y % 2 == 0):
            value += spin[z, x, bvc(y+1)]
            value += spin[z, x-1, bvc(y+1)]
            value += spin[z, x, y-1]
            value += spin[z, x-1, y-1]
        else:
            value += spin[z, bvc(x+1), bvc(y+1)]
            value += spin[z, x, bvc(y+1)]
            value += spin[z, bvc(x+1), y-1]
            value += spin[z, x, y-1]

        value += spin[z, bvc(x+1), y]
        value += spin[z, x-1, y]
        return value

    def calc():
        probs = random_uniform()
        if (probs < math.exp(2*J[0]*spin[z, x, y]*sum_nn()/T[n,l])):
            spin[z, x, y] *= np.int8(-1)

    if (p == 2 and q == 0) or (p == 0 and q == 1):
        calc()

import math
import numpy as np
from numba import cuda

@cuda.jit
def update_black_ext (spin, seed, T, J, h):
    m = T.shape[1]
    z, x, y = cuda.grid(3)
    n = int(math.floor (z / m))
    l = z % m
    p, q = x % 2, y % 2

    def random_uniform():
        seed[z, x, y] = np.int32((seed[z ,x, y]*1664525 + 1013904223) % 2**31)
        return seed[z, x, y] / (2**31)

    def bvc (x):
        if x == spin.shape[1]:
            x = 0
        return x       

    def sum_nn():
        value = 0
        value += spin[z, x, bvc(y+1)]
        value += spin[z, x, y-1]
        value += spin[z, bvc(x+1), y]
        value += spin[z, x-1, y]
        return value

    def calc():
        probs = random_uniform()
        if (probs < math.exp(2*(J[0]*sum_nn() - h[0])*spin[z, x, y]/T[n,l])):
            spin[z, x, y] *= np.int8(-1)

    if (p == 0 and q == 0) or (p == 1 and q == 1):
        calc()

@cuda.jit
def update_white_ext (spin, seed, T, J, h):
    m = T.shape[1]
    z, x, y = cuda.grid(3)
    n = int(math.floor (z / m))
    l = z % m
    p, q = x % 2, y % 2

    def random_uniform():
        seed[z, x, y] = np.int32((seed[z ,x, y]*1664525 + 1013904223) % 2**31)
        return seed[z, x, y] / (2**31)

    def bvc (x):
        if x == spin.shape[1]:
            x = 0
        return x       

    def sum_nn():
        value = 0
        value += spin[z, x, bvc(y+1)]
        value += spin[z, x, y-1]
        value += spin[z, bvc(x+1), y]
        value += spin[z, x-1, y]
        return value

    def calc():
        probs = random_uniform()
        if (probs < math.exp(2*(J[0]*sum_nn()-h[0])*spin[z, x, y]/T[n,l])):
            spin[z, x, y] *= np.int8(-1)

    if (p == 0 and q == 1) or (p == 1 and q == 0):
        calc()