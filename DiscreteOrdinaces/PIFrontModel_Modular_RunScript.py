# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:08:01 2024



@author: jzola2
"""

from PIFrontModel_1DGalerkin_Modular import PIFront
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
N = 8            # Number of discrete ordinances
M = 100          # Number of spatial cells in L
P = 2            # Number of flux basis functions
S = 4            # Number of ionization basis functions

L = 0.7          # Duct length [cm]
R = 0.1          # Duct radius [cm]
T_s = 0.1        # Source temperature [keV]
dt = 0.001       # Time step [ns]
t_max = 3.5      # End time [ns]
refl = 0         # Wall reflection coefficient

def alpha(r, phi, basis, R):
    u = 3*np.pi*(9*np.pi**2 - 64)**(-0.5)/R
    v = 8*R/(3*np.pi)
    
    q = 8*R*(9*np.pi/5*(9*np.pi**2 - 64)**(-1) - 2/(3*np.pi))
    p = R**(-2)*(1 - 576/25*(9*np.pi**2 - 64)**(-1))**(-0.5)
    
    theta = 0
    xdotomega = r*np.cos(theta)*np.cos(phi) + r*np.sin(theta)*np.sin(phi)
    D = xdotomega + ((xdotomega)**2 + R**2 - (r*np.cos(theta))**2 - (r*np.sin(theta))**2)**(0.5)
    
    if basis == 1:
        return 1
    elif basis == 2:
        return u*(D - v)
    else:
        return p*((D - v)*(D - v - q) - 1/(u**2))
    
def beta(r, phi, basis, R):
    if basis == 1:
        return np.sqrt(2)/R
    elif basis == 2:
        return 2*np.sqrt(6/R**6)*(r**2 - R**2/2)
    elif basis == 3:
        return np.sqrt(10/R**10)*(6*r**4 - 6*r**2*R**2 + R**4)
    elif basis == 4:
        return np.sqrt(14/R**14)*(-20*r**6 + 30*r**4*R**2 - 12*r**2*R**4 + R**6)
    elif basis == 5:
        return np.sqrt(18/R**18)*(70*r**8 - 140*r**6*R**2 + 90*r**4*R**4 - 20*r**2*R**6 + R**8)

def gaussian(r):
    std_dev = R/3
    
    return 1/(std_dev*(2*np.pi)**(0.5))*np.exp(-r**2/(2*std_dev**2))

def step(r):
    r0 = 0.75*R
    
    return (r <= r0)*1
    
def const(r):
    return r*0 + 1

# Define PIFront object
front = PIFront(N, M, P, S, L, R, refl, dt, t_max)

front.compute_basis_integrals(lambda r, phi, basis : alpha(r, phi, basis, R), lambda r, phi, basis : beta(r, phi, basis, R), quadriture='gauss')

front.set_boundary_condition(lambda r, phi, basis : alpha(r, phi, basis, R), T_s, step, pltit=False)

front.time_step(1e-8, 50, 500)

plt.title('Contours of Ionization Fraction over Time in ns')
plt.show()

# front_array = [PIFront(N, M, P, S, L, 0.1*b, refl, dt, t_max) for b in range(1, 6)]
# for i in range(5):
#     front_array[i].compute_basis_integrals(lambda r, phi, basis : alpha(r, phi, basis, (i + 1)*0.1), lambda r, phi, basis : beta(r, phi, basis, (i + 1)*0.1), quadriture='gauss')
    
#     front_array[i].set_boundary_condition(lambda r, phi, basis : alpha(r, phi, basis, (i + 1)*0.1), T_s, const, pltit=False)
    
#     front_array[i].time_step(1e-8, 50, 500)
    
# for i in range(5):
#     front_array[i].plot_front_location()  
# plt.legend()
# plt.show()