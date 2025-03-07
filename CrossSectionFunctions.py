# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:34:29 2025

@author: jzola2
"""

import numpy as np

def blackbody(nu, T):
    h = 4.135e-15   # Planck's constant [eV-s]
    c = 30          # Speed of light [cm/ns]

    return 2*h*nu**3/(c**2)*1/(np.exp(h*nu/T) - 1)

def pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E):
    # The fit equation for photoionization cross section
    x = E/E0 - y0
    
    y = np.sqrt(x**2 + y1**2)
    
    mb2cm2 = 1e-18 # Conversion factor from Mb to cm^2
    
    return sigma0*((x - 1)**2 + yw**2)*y**(0.5*P - 5.5)*(1 + np.sqrt(y/ya))**(-P)*mb2cm2

def pi_n1(E):
    E0 = 1.240
    sigma0 = 1.745e3
    ya = 3.784
    P = 17.64
    yw = 7.589e-2
    y0 = 8.698
    y1 = 0.1271
    
    return pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)

def pi_n2(E):
    E0 = 1.386
    sigma0 = 59.67
    ya = 31.75
    P = 8.943
    yw = 1.934e-2
    y0 = 21.31
    y1 = 1.503e-2
    
    return pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)

def pi_n3(E):
    E0 = 0.1723
    sigma0 = 6.753e2
    ya = 3.852e2
    P = 6.822
    yw = 0.1191
    y0 = 3.839e-3
    y1 = 0.4569
    
    return pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)
    
def pi_n4(E):
    E0 = 0.2044
    sigma0 = 0.8659
    ya = 4.931e2
    P = 8.785
    yw = 3.143
    y0 = 3.328e2
    y1 = 42.85
    
    return pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)
    
def pi_n5(E):
    E0 = 7.824
    sigma0 = 68.64
    ya = 32.10
    P = 5.495
    yw = 0
    y0 = 0
    y1 = 0
    
    return pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)

def rrxs_fitequation(A, B, T0, T1, T, **kwargs):
    # The fit equation for radiative recombination
    for key, value in kwargs.items():
        if key == 'C':
            C = value
        elif key == 'T2':
            T2 = value
            
    if len(kwargs) > 0:
        B += C*np.exp(-T2/T)
        
    is2ins = 1e-9 # Conversion factor from 1/s to 1/ns
        
    return is2ins*A/(np.sqrt(T/T0)*(1 + np.sqrt(T/T0))**(1 - B)*(1 + np.sqrt(T/T1))**(1 + B))

def rr_n1(T):
    k = 8.6133e-5 # Boltzmann constant in eV/K
    A = 6.622e-11
    B = 0.6109
    T0 = 4.136*k
    T1 = 4.216e6*k
    C = 0.4093
    T2 = 8.770e4*k
    
    return rrxs_fitequation(A, B, T0, T1, T, C=C, T2=T2)

def rr_n2(T):
    k = 8.6133e-5 # Boltzmann constant in eV/K
    A = 2.096e-9
    B = 0.7668
    T0 = 0.1602*k
    T1 = 4.377e6*k
    C = 0.1070
    T2 = 1.392e5*k
    
    return rrxs_fitequation(A, B, T0, T1, T, C=C, T2=T2)

def rr_n3(T):
    k = 8.6133e-5 # Boltzmann constant in eV/K
    A = 2.501e-9
    B = 0.7844
    T0 = 0.5235*k
    T1 = 4.470e6*k
    C = 0.0447
    T2 = 1.642e5*k
    
    return rrxs_fitequation(A, B, T0, T1, T, C=C, T2=T2)

def rr_n4(T):
    k = 8.6133e-5 # Boltzmann constant in eV/K
    A = 3.955e-9
    B = 0.7813
    T0 = 0.6821*k
    T1 = 5.076e6*k
    
    return rrxs_fitequation(A, B, T0, T1, T)

def rr_n5(T):
    k = 8.6133e-5 # Boltzmann constant in eV/K
    A = 1.724e-10
    B = 0.6556
    T0 = 3.372e2*k
    T1 = 1.030e7*k
    
    return rrxs_fitequation(A, B, T0, T1, T)