import numpy as np
from numba import vectorize, float64
import math

@vectorize([float64(float64)])
def exp1(x):
    max_it = 100
    EU_GAMMA = 0.577215664901533
    alpha = 1e-30
    eps = 1e-15

    if x > 1:
        # Evaluate exponential integral using continued fraction
        a = lambda n : 1*(n == 1) - (n - 1)**2
        b = lambda n : x + 2*n - 1
        b0 = 0
        f0 = b0*(b0 != 0) + alpha*(b0 == 0)
        C0 = f0
        D0 = 0

        C = b(1) + a(1)/C0
        D = b(1) + a(1)*D0

        if D == 0:
            D = alpha
        else:
            D = D**-1

        if C == 0:
            C = alpha

        delta = C*D

        f = f0*delta

        index = 2
        while (abs(delta - 1) > eps and index < max_it):
            an = a(index)
            bn = b(index)

            C = bn + an/C
            D = bn + an*D

            if D == 0:
                D = alpha
            else:
                D = D**-1

            if C == 0:
                C = alpha

            delta = C*D
            f = f*delta

            index += 1

        return np.exp(-x)*f
    else:
        # Evaluate exponential integral using series representation
        f = -np.log(x) - EU_GAMMA
        err = -x
        index = 2
        fact = 1
        while abs(err) > eps and index <= max_it:
            f -= err
            fact *= index
            err = (-x)**(index)/((index)*fact)

            index += 1
        return f