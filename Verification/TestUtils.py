from Utils.Utils import exp1
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Exponential integral comparison test

n_points = 1_000_000
x_max = 100
x = np.linspace(0, x_max, n_points)

y_scipy = special.exp1(x)
y_utils = exp1(x)

plt.plot(x, y_utils, label='Utils')
plt.plot(x, y_scipy, label='Scipy')
plt.legend()
plt.show()