# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:34:29 2025

@author: jzola2
"""

import numpy as np
import scipy
from numba import jit, int32, float64
from numba.experimental import jitclass
import matplotlib.pyplot as plt

#@jit
def blackbody(nu_list, T):
    """
    Parameters
    ----------
    nu : double
        The photon frequency at which to evaluate the blackbody distribution in
        GHz
    T : double
        The temperature of the blackbody emitter in keV

    Returns
    -------
    e : double
        The energy density in [keV/cm^2] of photons of energy h*nu for a blackbody 
        emitter at temperature T
    """
    h = 4.135e-9     # Planck's constant [keV-ns]
    c = 30.0         # Speed of light [cm/ns]

    if hasattr(nu_list, "__len__"):
        e = np.zeros((len(nu_list), ))

        for i, nu in enumerate(nu_list):
            if h*nu/T > 600:
                e[i] = 0.0
            else:
                e[i] = 2*h*nu**3/(c**2)*(np.exp(h*nu/T) - 1)**-1     # Intensity [keV/cm^2]
    else:
        if h*nu_list/T > 600:
            e = np.array([0.0])
        else:
            e = np.array([2*h*nu_list**3/(c**2)*(np.exp(h*nu_list/T) - 1)**-1])

    return e

#@jit
def sample_blackbody(T, xi):
    a = 0.01372         # Source spectrum [GJ/(cm^3*keV^4)]
    c = 30              # Speed of light [cm/ns]
    keV2GJ = 1.602e-25  # Conversion factor for keV to GJ
    h = 4.135e-9        # Planck's constant [keV-ns]
    
    n = 1.0
    const = np.pi**4/90*xi[0]
    comp = 1.0

    while const > comp:
        n += 1.0
        comp += 1/n**4

    return -1/n*np.log(np.prod(xi[1:]))*T/h

    #nu0 = 2.71*T/h
    #sol = scipy.optimize.root(lambda nu : 4*np.pi*scipy.integrate.quad(lambda nu_prime : blackbody(nu_prime, T), 1e6, nu)[0] - (a*c*T**4/keV2GJ)*xi, nu0, jac=lambda nu : 4*np.pi*blackbody(nu, T))
    #return sol.x[0]

def integrate_planck_in_energy(hnu_low, hnu_high, T):
    N_terms = 12
    x_low = hnu_low/T
    x_high = hnu_high/T
    c = 30                  # Speed of light [cm/ns]
    h = 4.135e-9            # Planck's constant [keV-ns]
    nu_low = hnu_low/h
    nu_high = hnu_high/h

    flux = 0

    for n in range(1, N_terms + 1):
        nk = n*h/T
        flux += np.exp(-n*x_high)*(-nu_high**3/nk - 3*nu_high**2/nk**2 - 3*nu_high/nk**3 - 6/nk**4) \
                + np.exp(-n*x_low)*(nu_low**3/nk + 3*nu_low**2/nk**2 + 3*nu_low/nk**3 + 6/nk**4)
        
    return 2*h/c**2*flux

def sample_maxwellian(T, xi):
    return 1.5*T

spec = [
    ('Z', int32),
    ('Eth', float64[:]),
    ('Emax', float64[:]),
]

#@jitclass(spec)
class Nitrogen():
    def __init__(self):
        self.Z = 7
        self.Eth = np.zeros((self.Z, ))
        self.Emax = np.zeros((self.Z, ))

        # Threshhold energies in keV
        self.Eth[0] = 0.01453
        self.Eth[1] = 0.02960
        self.Eth[2] = 0.04745
        self.Eth[3] = 0.07747
        self.Eth[4] = 0.09789
        self.Eth[5] = 0.5521
        self.Eth[6] = 0.6671

        # Maximum photoionization energy fits in keV
        self.Emax[0] = 0.4048
        self.Emax[1] = 0.4236
        self.Emax[2] = 0.4473
        self.Emax[3] = 0.4753
        self.Emax[4] = 0.5043
        self.Emax[5] = 50.000
        self.Emax[6] = 50.000

    def pi_n(self, E, level):
        match level:
            case 0:
                return self.pi_n1(E)
            case 1:
                return self.pi_n2(E)
            case 2:
                return self.pi_n3(E)
            case 3:
                return self.pi_n4(E)
            case 4:
                return self.pi_n5(E)
            case 5:
                return self.pi_n6(E)
            case 6:
                return self.pi_n7(E)

    def rr_n(self, T, level):
        match level:
            case 1:
                return self.rr_n1(T)
            case 2:
                return self.rr_n2(T)
            case 3:
                return self.rr_n3(T)
            case 4:
                return self.rr_n4(T)
            case 5:
                return self.rr_n5(T)
            case 6:
                return self.rr_n6(T)
            case 7: 
                return self.rr_n7(T)
            
    def sigma_n(self, T, level):
        match level:
            case 1:
                return self.sigma_n1(T)
            case 2:
                return self.sigma_n2(T)
            case 3:
                return self.sigma_n3(T)
            case 4:
                return self.sigma_n4(T)
            case 5:
                return self.sigma_n5(T)
            case 6:
                return self.sigma_n6(T)
            case 7:
                return self.sigma_n7(T)

    def pixs_fitequation(self, E0, sigma0, ya, P, yw, y0, y1, E):
        # The fit equation for photoionization cross section
        x = E/E0 - y0
    
        y = np.sqrt(x**2 + y1**2)
    
        mb2cm2 = 1e-18 # Conversion factor from Mb to cm^2
    
        return sigma0*((x - 1)**2 + yw**2)*y**(0.5*P - 5.5)*(1 + np.sqrt(y/ya))**(-P)*mb2cm2

    def pi_n1(self, E):
        # Input - energy in keV
        E0 = 4.034e-3
        sigma0 = 8.235e2
        ya = 80.33
        P = 3.928
        yw = 9.097e-2
        y0 = 0.8598
        y1 = 2.325
    
        return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)

    def pi_n2(self, E):
        # Input - energy in keV
        E0 = 6.128e-5
        sigma0 = 1.944
        ya = 816.3
        P = 8.773
        yw = 10.43
        y0 = 428
        y1 = 20.30
    
        return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)

    def pi_n3(self, E):
        # Input - energy in keV
        E0 = 0.2420e-3
        sigma0 = 0.9375
        ya = 278.8
        P = 9.156
        yw = 1.850
        y0 = 187.7
        y1 = 3.999
    
        return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)
    
    def pi_n4(self, E):
        # Input - energy in keV
        E0 = 5.494e-3
        sigma0 = 1.690e4
        ya = 1.714
        P = 17.06
        yw = 7.904
        y0 = 6.415e-3
        y1 = 1.937e-2
    
        return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)
    
    def pi_n5(self, E):
        # Input - energy in keV
        E0 = 4.471e-3
        sigma0 = 83.76
        ya = 32.97
        P = 6.003
        yw = 0
        y0 = 0
        y1 = 0
    
        return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)
    
    def pi_n6(self, E):
        # Input - energy in keV
        E0 = 69.43e-3
        sigma0 = 151.9
        ya = 26.27
        P = 2.315
        yw = 0
        y0 = 0
        y1 = 0
    
        return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)
    
    def pi_n7(self, E):
        # Input - energy in keV
        E0 = 21.8e-3
        sigma0 = 1.117e3
        ya = 32.88
        P = 2.693
        yw = 0
        y0 = 0
        y1 = 0
    
        return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E)

    def rrxs_fitequation(self, A, B, T0, T1, T, **kwargs):
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

    def rr_n1(self, T):
        k = 8.6133e-8 # Boltzmann constant in keV/K
        A = 6.387e-10
        B = 0.7308
        T0 = 9.467e-2*k
        T1 = 2.954e6*k
        C = 0.2440
        T2 = 6.739e4*k
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C=C, T2=T2)
    
    def rr_n2(self, T):
        k = 8.6133e-8 # Boltzmann constant in keV/K
        A = 2.410e-9
        B = 0.7948
        T0 = 0.1231*k
        T1 = 3.016e6*k
        C = 0.0774
        T2 = 1.106e5*k
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C=C, T2=T2)

    def rr_n3(self, T):
        k = 8.6133e-8 # Boltzmann constant in keV/K
        A = 7.923e-10
        B = 0.7768
        T0 = 3.750*k
        T1 = 3.468e6*k
        C = 0.0223
        T2 = 7.206e4
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C=C, T2=T2)

    def rr_n4(self, T):
        k = 8.6133e-8 # Boltzmann constant in keV/K
        A = 1.553e-10
        B = 0.6682
        T0 = 1.823e2*k
        T1 = 7.751e6*k
    
        return self.rrxs_fitequation(A, B, T0, T1, T)
    
    def rr_n5(self, T):
        k = 8.6133e-8 # Boltzmann constant in keV/K
        A = 6.245e-11
        B = 0.4985
        T0 = 1.957e3*k
        T1 = 2.177e7*k
    
        return self.rrxs_fitequation(A, B, T0, T1, T)
    
    def rr_n6(self, T):
        k = 8.6133e-8 # Boltzmann constant in keV/K
        A = 2.388e-10
        B = 0.6732
        T0 = 53.960e2*k
        T1 = 3.583e7*k
    
        return self.rrxs_fitequation(A, B, T0, T1, T)
    
    def rr_n7(self, T):
        k = 8.6133e-8 # Boltzmann constant in keV/K
        A = 6.170e-10
        B = 0.7470
        T0 = 1.951e2*k
        T1 = 4.483e7*k
    
        return self.rrxs_fitequation(A, B, T0, T1, T)

    def sigma_fitequation(self, dE, P, A, X, k, Te):
        U = dE/Te 

        return A*(1 + P*U**0.5)/(X + U)*U**k*np.exp(-U)

    def sigma_n1(self, Te):
        dE = 14.5e-3    # Ionization energy in [keV]
        P = 0           # 
        A = 0.482e-7    # Rate coefficient [cm^3/s]
        X = 0.0652
        k = 0.42

        return self.sigma_fitequation(dE, P, A, X, k, Te)

    def sigma_n2(self, Te):
        dE = 29.6e-3    # Ionization energy in [keV]
        P = 0           # 
        A = 0.298e-7    # Rate coefficient [cm^3/s]
        X = 0.310
        k = 0.30

        return self.sigma_fitequation(dE, P, A, X, k, Te)

    def sigma_n3(self, Te):
        dE = 47.5e-3    # Ionization energy in [keV]
        P = 1           # 
        A = 0.810e-8    # Rate coefficient [cm^3/s]
        X = 0.350
        k = 0.24

        return self.sigma_fitequation(dE, P, A, X, k, Te)

    def sigma_n4(self, Te):
        dE = 77.5e-3    # Ionization energy in [keV]
        P = 1           # 
        A = 0.371e-8    # Rate coefficient [cm^3/s]
        X = 0.549
        k = 0.18

        return self.sigma_fitequation(dE, P, A, X, k, Te)

    def sigma_n5(self, Te):
        dE = 97.9e-3    # Ionization energy in [keV]
        P = 0           # 
        A = 0.151e-8    # Rate coefficient [cm^3/s]
        X = 0.0167
        k = 0.74

        return self.sigma_fitequation(dE, P, A, X, k, Te)
    
    def sigma_n6(self, Te):
        dE = 552.1e-3
        P = 0
        A = 0.371e-9
        X = 0.546
        k = 0.16

        return self.sigma_fitequation(dE, P, A, X, k, Te)
    
    def sigma_n7(self, Te):
        dE = 667.0e-3
        P = 1
        A = 0.777e-10
        X = 0.624
        k = 0.16

        return self.sigma_fitequation(dE, P, A, X, k, Te)

    def E_spectral(self, Te, ni):
        h = 4.135e-9        # Planck's constant [keV-ns]
        a = 0.01372
        c = 30.0            # Speed of light [cm/s]
        keV2GJ = 1.602e-25  # Conversion factor for keV to GJ

        levels = len(ni)
        ne = np.dot(np.arange(1, levels), ni[1:])

        Esp = 0.0

        for i in range(levels - 1):
            Gamma = scipy.integrate.quad(lambda nu : blackbody(nu, Te)*self.pi_n(h*nu, i)*(h*nu - self.Eth[i]), self.Eth[i]/h, self.Emax[i]/h)[0]/(a*c*Te**4/(4*np.pi*keV2GJ))
            flux = scipy.integrate.quad(lambda nu : blackbody(nu, Te)/(h*nu), self.Eth[i]/h, self.Emax[i]/h)[0]

            R = self.rr_n(Te, i + 1)*(1.5*Te + self.Eth[i])

            Esp += Gamma*flux*ni[i] - R*ne*ni[i + 1]

        return Esp

#@jit
def cv(T, rho):
    return 1.5*rho

# Test sample blackbody and blackbody
#N = 1000000
#a = 0.01372
#c = 30.0
#keV2GJ = 1.602e-25
#T = 1.0
#h = 4.135e-9
#bins = np.linspace(0, 5e9, 10000)
#bin_centers = bins[:-1] + np.diff(bins)
#bin_widths = np.diff(bins)
#heights = np.zeros((len(bin_centers), ))
#rng = np.random.default_rng()

#analytic = blackbody(bin_centers, T)/(a*c*T**4/(4*np.pi*keV2GJ))

#for i in range(N):
#    nu = sample_blackbody(T, rng.random(5))
#    if nu > bins[-1]:
#        index = len(bin_centers) - 1
#    else:
#        index = np.searchsorted(bins, nu) - 1
#    heights[index] += 1/(bin_widths[index]*N)

#lt.bar(bin_centers, heights, bin_widths[0], align='center')
#plt.plot(bin_centers, analytic, 'tab:orange', label='Analytic')
#plt.legend()
#plt.show()