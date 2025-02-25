# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:45:24 2025

This program models the propogation of a photoionization front through a circular
jet of gas

@author: jzola2
"""

import numpy as np
import matplotlib.pyplot as plt

tol = 1e-6
max_it = 50
x = 1
R = 0.5
Z = 1

def nonlinear_solve(theta_guess, tol, max_it):
    it = 0
    theta_old = theta_guess[0]
    err =  -1/np.tan(theta_old) - (x - R*np.sin(theta_old))/(Z - R*np.cos(theta_old))
    err_old = err

    theta = theta_guess[1]

    while abs(err) > tol and it < max_it:
        err =  -1/(np.tan(theta)) - (x - R*np.sin(theta))/(Z - R*np.cos(theta))

        temp = theta - (theta - theta_old)/(err - err_old)*err
    
        theta_old = theta
        theta = temp
    
        err_old = err
    
        it += 1
        
    return theta, err

theta1, err1 = nonlinear_solve((-np.pi/24, -np.pi/6), tol, max_it)
theta2, err2 = nonlinear_solve((np.pi/2, 5*np.pi/6), tol, max_it)

phi = np.linspace(0, 2*np.pi, 10000)
fig, ax = plt.subplots(1, 1)
ax.plot(Z - R*np.cos(phi), R*np.sin(phi))
ax.plot([0, 0], [-1, 1])
ax.plot([0, Z - R*np.cos(theta1)], [1, R*np.sin(theta1)], 'g--')
ax.plot([0, Z - R*np.cos(theta2)], [1, R*np.sin(theta2)], 'g--')
ax.axis('equal')
plt.show()

theta = np.linspace(0, abs(theta1), 10000)
phi_plus = np.arctan((x + R*np.sin(theta))/(Z - R*np.cos(theta)))
phi_minus = -np.arctan((x - R*np.sin(theta))/(Z - R*np.cos(theta)))

theta_b = np.linspace(abs(theta1), abs(theta2), 10000)
x1 = (R*np.cos(theta_b) - Z)/np.tan(theta_b) + R*np.sin(theta_b)
phi_plus_b = np.arctan((R*np.sin(theta_b) - x1)/(Z - R*np.cos(theta_b)))
phi_minus_b = -np.arctan((x - R*np.sin(theta_b))/(Z - R*np.cos(theta_b)))

plt.plot(np.concatenate((theta, theta_b)), np.concatenate((phi_plus, phi_plus_b)), label='$\phi^+$')
plt.plot(np.concatenate((theta, theta_b)), np.concatenate((phi_minus, phi_minus_b)), label='$\phi^-$')
plt.legend()
plt.xlabel('$\\theta$')
plt.ylabel('$\\phi$')
plt.show()