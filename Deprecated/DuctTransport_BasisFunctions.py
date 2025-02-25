# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:54:04 2024

This script describes the basis functions for transport in ducts as developed 
in Garcia, Ono and Viera

@author: jzola2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

R = 0.1

def alpha1(r, phi):
    return 1

def alpha2(r, phi):
    u = 3*np.pi*(9*np.pi**2 - 64)**(-0.5)/R
    v = 8*R/(3*np.pi)
    theta = 0
    xdotomega = r*np.cos(theta)*np.cos(phi) + r*np.sin(theta)*np.sin(phi)
    D = xdotomega + ((xdotomega)**2 + R**2 - (r*np.cos(theta))**2 - (r*np.sin(theta))**2)**(0.5)
    
    return u*(D - v)

def alpha3(r, phi):
    u = 3*np.pi*(9*np.pi**2 - 64)**(-0.5)/R
    v = 8*R/(3*np.pi)
    
    q = 8*R*(9*np.pi/5*(9*np.pi**2 - 64)**(-1) - 2/(3*np.pi))
    p = R**(-2)*(1 - 576/25*(9*np.pi**2 - 64)**(-1))**(-0.5)
    
    theta = 0
    xdotomega = r*np.cos(theta)*np.cos(phi) + r*np.sin(theta)*np.sin(phi)
    D = xdotomega + ((xdotomega)**2 + R**2 - (r*np.cos(theta))**2 - (r*np.sin(theta))**2)**(0.5)
    
    return p*((D - v)*(D - v - q) - 1/(u**2))

#r_cell = np.linspace(0, R, int(1e5))
N = 1000
x = np.linspace(-R, R, N)
y = np.linspace(-R, R, N)

basis1 = np.zeros((N, N))
basis2 = np.zeros((N, N))
basis3 = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if (x[i]**2 + y[j]**2 <= R**2):
            r_cell = (x[i]**2 + y[j]**2)**0.5
            phi_cell = np.tan(y[j]/x[i])
            basis1[i, j] = alpha1(r_cell, phi_cell)
            basis2[i, j] = alpha2(r_cell, phi_cell)
            basis3[i, j] = alpha3(r_cell, phi_cell)

r_edge = np.linspace(0, R, N)
basis4 = np.zeros((N, ))
basis5 = np.zeros((N, ))
basis6 = np.zeros((N, ))

for i in range(N):
    basis4[i] = 1/(2*np.pi)*integrate.quad(lambda phi : alpha1(r_edge[i], phi), 0, 2*np.pi)[0]
    basis5[i] = 1/(2*np.pi)*integrate.quad(lambda phi : alpha2(r_edge[i], phi), 0, 2*np.pi)[0]
    basis6[i] = 1/(2*np.pi)*integrate.quad(lambda phi : alpha3(r_edge[i], phi), 0, 2*np.pi)[0]

fig, ax = plt.subplots(3, 1)
ax1 = plt.subplot(3, 1, 1)
ax1.pcolormesh(x, y, basis1)
ax1.axis('equal')
ax2 = plt.subplot(3, 1, 2)
ax2.pcolormesh(x, y, basis2)
ax2.axis('equal')
ax3 = plt.subplot(3, 1, 3)
ax3.pcolormesh(x, y, basis3)
ax3.axis('equal')
plt.show()

plt.plot(r_edge, basis4, label='1')
plt.plot(r_edge, basis5, label='2')
plt.plot(r_edge, basis6, label='3')
plt.xlabel('Radius')
plt.ylabel('Basis function value')
plt.legend()
plt.show()