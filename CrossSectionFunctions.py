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
import Constants
import math

@jit
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

    if hasattr(nu_list, "__len__"):
        e = np.zeros((len(nu_list), ))

        for i, nu in enumerate(nu_list):
            if Constants.h*nu/T > 600:
                e[i] = 0.0
            else:
                e[i] = 2*Constants.h*nu**3/(Constants.c**2)*(np.exp(Constants.h*nu/T) - 1)**-1     # Intensity [keV/cm^2]
    else:
        if Constants.h*nu_list/T > 600:
            e = np.array([0.0])
        else:
            e = np.array([2*Constants.h*nu_list**3/(Constants.c**2)*(np.exp(Constants.h*nu_list/T) - 1)**-1])

    return e

@jit
def sample_blackbody(T, xi):
    n = 1.0
    const = np.pi**4/90*xi[0]
    comp = 1.0

    while const > comp:
        n += 1.0
        comp += 1/n**4

    return -1/n*np.log(np.prod(xi[1:]))*T/Constants.h

@jit
def integrate_planck_in_energy(hnu_low, hnu_high, T):
    N_terms = 12
    x_low = hnu_low/T
    x_high = hnu_high/T
    nu_low = hnu_low/Constants.h
    nu_high = hnu_high/Constants.h

    flux = 0

    for n in range(1, N_terms + 1):
        nk = n*Constants.h/T
        flux += np.exp(-n*x_high)*(-nu_high**3/nk - 3*nu_high**2/nk**2 - 3*nu_high/nk**3 - 6/nk**4) \
                + np.exp(-n*x_low)*(nu_low**3/nk + 3*nu_low**2/nk**2 + 3*nu_low/nk**3 + 6/nk**4)
        
    return 2*Constants.h/Constants.c**2*flux

@jit
def integrate_planck_in_number(hnu_low, hnu_high, T):
    N_terms = 12
    x_low = hnu_low/T
    x_high = hnu_high/T
    nu_low = hnu_low/Constants.h
    nu_high = hnu_high/Constants.h

    flux = 0

    for n in range(1, N_terms + 1):
        nk = n*Constants.h/T
        flux += np.exp(-n*x_low)*(nu_low**2/nk + 2*nu_low/(nk**2) + 2/(nk**3)) \
              - np.exp(-n*x_high)*(nu_high**2/nk + 2*nu_high/(nk**2) + 2/(nk**3))

    return 2/(Constants.c**2)*flux

@jit
def sample_maxwellian(T, xi):
    return 1.5*T

spec = [
    ('Z', int32),
    ('Eth', float64[:]),
    ('Emax', float64[:]),
]

@jitclass(spec)
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
        if E < self.Eth[level]:
            return 0.0

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
            case 0:
                return self.sigma_n1(T)
            case 1:
                return self.sigma_n2(T)
            case 2:
                return self.sigma_n3(T)
            case 3:
                return self.sigma_n4(T)
            case 4:
                return self.sigma_n5(T)
            case 5:
                return self.sigma_n6(T)
            case 6:
                return self.sigma_n7(T)

    def tbr_n(self, T, level):
        sigma_eii = self.sigma_n(T, level - 1)

        #if T < 0.01:
        #    tbr = self.sigma_n(0.01, level - 1)*0.5*((Constants.h*Constants.c)**2/(2*np.pi*Constants.me_keV*T))**1.5*np.exp(self.Eth[level - 1]/0.01)
        #else:
        #    tbr = sigma_eii*0.5*((Constants.h*Constants.c)**2/(2*np.pi*Constants.me_keV*T))**1.5*np.exp(self.Eth[level - 1]/T)

        #log_space_tbr = np.log(sigma_eii*0.5*((Constants.h*Constants.c)**2/(2*np.pi*Constants.me_keV*T))**1.5)
        log_space_tbr = np.log(sigma_eii*0.5) + 3*np.log(Constants.h*Constants.c) - 1.5*np.log(2*np.pi*Constants.me_keV*T)
        log_space_tbr += (self.Eth[level] - self.Eth[level - 1])/T

        #tbr = np.exp(log_space_tbr)
#
        #if math.isinf(tbr):
        #    print("Infinite tbr reached")
        #    print("In log space: ")
        #    print(log_space_tbr)
        #    print("EII: ")
        #    print(sigma_eii)
        #    print("Ratio of Threshhold energy to temp")
        #    print((self.Eth[level] - self.Eth[level - 1])/T)
        #    print("Temperature: ")
        #    print(T)
        #    print("Ionization Level: ")
        #    print(level)
        #    assert not math.isinf(tbr)

        return log_space_tbr
        #return tbr

    def nl2ind(self, n, l):
        return n + l - 1

    def pixs_fitequation(self, E0, sigma0, ya, P, yw, y0, y1, E, subshell):
        # The fit equation for photoionization cross section
        x = E/E0 - y0
    
        y = np.sqrt(x**2 + y1**2)
    
        mb2cm2 = 1e-18 # Conversion factor from Mb to cm^2

        Q = 0.5*P - 5.5 - subshell
    
        return sigma0*((x - 1)**2 + yw**2)*y**(Q)*(1 + np.sqrt(y/ya))**(-P)*mb2cm2

    def pi_n1(self, E):
        # Input - energy in keV

        if E < self.Emax[0]:
            E0 = 4.034e-3
            sigma0 = 8.235e2
            ya = 80.33
            P = 3.928
            yw = 9.097e-2
            y0 = 0.8598
            y1 = 2.325
    
            return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E, 0)
        else:
            pixs_sum = 0.0
            Eth_subshell = np.array([0.4048, 0.02541, 0.01453])
            E0_subshell = np.array([0.1270, 0.01482, 0.01164])
            sigma0_subshell = np.array([47.48, 772.2, 0.1029e5])
            ya_subshell = np.array([138.0, 2.306, 2.361])
            P_subshell = np.array([1.252, 9.139, 8.821])
            yw_subshell = np.array([0.0, 0.0, 0.4239])
            for n in range(1, 3):
                for l in range(n):
                    ind = self.nl2ind(n, l)
                    if E >= Eth_subshell[ind]:
                        pixs_sum += self.pixs_fitequation(E0_subshell[ind], sigma0_subshell[ind], ya_subshell[ind], P_subshell[ind], yw_subshell[ind], 0, 0, E, l)

            return pixs_sum

    def pi_n2(self, E):
        # Input - energy in keV

        if E < self.Emax[1]:
            E0 = 6.128e-5
            sigma0 = 1.944
            ya = 816.3
            P = 8.773
            yw = 10.43
            y0 = 428
            y1 = 20.30
    
            return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E, 0)
        else:
            pixs_sum = 0.0
            Eth_subshell = np.array([0.4236, 0.03796, 0.02960])
            E0_subshell = np.array([0.1242, 0.01094, 0.01827])
            sigma0_subshell = np.array([50.02, 0.7483e3, 0.1724e3])
            ya_subshell = np.array([91.00, 2.793, 88.93])
            P_subshell = np.array([1.335, 9.956, 3.348])
            yw_subshell = np.array([0.0, 0.0, 0.4209])
            for n in range(1, 3):
                for l in range(n):
                    ind = self.nl2ind(n, l)
                    if E >= Eth_subshell[ind]:
                        pixs_sum += self.pixs_fitequation(E0_subshell[ind], sigma0_subshell[ind], ya_subshell[ind], P_subshell[ind], yw_subshell[ind], 0, 0, E, l)

            return pixs_sum

    def pi_n3(self, E):
        # Input - energy in keV

        if E < self.Emax[2]:
            E0 = 0.2420e-3
            sigma0 = 0.9375
            ya = 278.8
            P = 9.156
            yw = 1.850
            y0 = 187.7
            y1 = 3.999

            return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E, 0)
        else:
            pixs_sum = 0.0
            Eth_subshell = np.array([0.4473, 0.05545, 0.04745])
            E0_subshell = np.array([0.1220, 5.853e-3, 0.01925])
            sigma0_subshell = np.array([52.35, 0.1908e3, 94.00])
            ya_subshell = np.array([94.28, 6.264, 0.1152e3])
            P_subshell = np.array([1.335, 9.711, 3.194])
            yw_subshell = np.array([0.0, 0.0, 0.5946])
            for n in range(1, 3):
                for l in range(n):
                    ind = self.nl2ind(n, l)
                    if E >= Eth_subshell[ind]:
                        pixs_sum += self.pixs_fitequation(E0_subshell[ind], sigma0_subshell[ind], ya_subshell[ind], P_subshell[ind], yw_subshell[ind], 0, 0, E, l)

            return pixs_sum
    
    def pi_n4(self, E):
        # Input - energy in keV

        if E < self.Emax[3]:
            E0 = 5.494e-3
            sigma0 = 1.690e4
            ya = 1.714
            P = 17.06
            yw = 7.904
            y0 = 6.415e-3
            y1 = 1.937e-2

            return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E, 0)
        else:
            pixs_sum = 0.0
            Eth_subshell = np.array([0.4753, 0.07747])
            E0_subshell = np.array([0.1070, 6.225e-3])
            sigma0_subshell = np.array([70.46, 0.1110e3])
            ya_subshell = np.array([53.42, 17.33])
            P_subshell = np.array([1.552, 6.719])
            yw_subshell = np.array([0.0, 0.0])
            for n in range(1, 3):
                l = 0
                ind = self.nl2ind(n, l)
                if E >= Eth_subshell[ind]:
                    pixs_sum += self.pixs_fitequation(E0_subshell[ind], sigma0_subshell[ind], ya_subshell[ind], P_subshell[ind], yw_subshell[ind], 0, 0, E, l)

            return pixs_sum
    
    def pi_n5(self, E):
        # Input - energy in keV

        if E < self.Emax[4]:
            E0 = 4.471e-3
            sigma0 = 83.76
            ya = 32.97
            P = 6.003
            yw = 0
            y0 = 0
            y1 = 0

            return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E, 0)
        else:
            pixs_sum = 0.0
            Eth_subshell = np.array([0.5043, 0.09789])
            E0_subshell = np.array([0.1060, 0.01862])
            sigma0_subshell = np.array([73.04, 34.47])
            ya_subshell = np.array([55.47, 42.31])
            P_subshell = np.array([1.528, 3.606])
            yw_subshell = np.array([0.0, 0.0])
            for n in range(1, 3):
                l = 0
                ind = self.nl2ind(n, l)
                if E >= Eth_subshell[ind]:
                    pixs_sum += self.pixs_fitequation(E0_subshell[ind], sigma0_subshell[ind], ya_subshell[ind], P_subshell[ind], yw_subshell[ind], 0, 0, E, l)

            return pixs_sum
    
    def pi_n6(self, E):
        # Input - energy in keV

        if E < self.Emax[5]:
            E0 = 69.43e-3
            sigma0 = 151.9
            ya = 26.27
            P = 2.315
            yw = 0
            y0 = 0
            y1 = 0

            return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E, 0)
        else:
            Eth_subshell = 0.5521
            E0 = 0.06943
            sigma0 = 0.1519e3
            ya = 26.27
            P = 2.315
            yw = 0
            y0 = 0
            y1 = 0
            
            if E >= Eth_subshell:
                return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E, 0)
            else:
                return 0.0
    
    def pi_n7(self, E):
        # Input - energy in keV

        if E < self.Emax[6]:
            E0 = 21.08e-3
            sigma0 = 1.117e3
            ya = 32.88
            P = 2.693
            yw = 0
            y0 = 0
            y1 = 0
    
            return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E, 0)
        else:
            Eth_subshell = 0.6671
            E0 = 21.08e-3
            sigma0 = 0.1117e4
            ya = 32.88
            P = 2.963
            yw = 0
            y0 = 0
            y1 = 0
            
            if E >= Eth_subshell:
                return self.pixs_fitequation(E0, sigma0, ya, P, yw, y0, y1, E, 0)
            else:
                return 0.0

    def rrxs_fitequation(self, A, B, T0, T1, T, C, T2):
        # The fit equation for radiative recombination
        B += C*np.exp(-T2/T)
        
        is2ins = 1e-9 # Conversion factor from 1/s to 1/ns
        
        return is2ins*A/(np.sqrt(T/T0)*(1 + np.sqrt(T/T0))**(1 - B)*(1 + np.sqrt(T/T1))**(1 + B))

    def rr_n1(self, T):
        A = 6.387e-10
        B = 0.7308
        T0 = 9.467e-2*Constants.k_B
        T1 = 2.954e6*Constants.k_B
        C = 0.2440
        T2 = 6.739e4*Constants.k_B
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C, T2)
    
    def rr_n2(self, T):
        A = 2.410e-9
        B = 0.7948
        T0 = 0.1231*Constants.k_B
        T1 = 3.016e6*Constants.k_B
        C = 0.0774
        T2 = 1.106e5*Constants.k_B
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C, T2)

    def rr_n3(self, T):
        A = 7.923e-10
        B = 0.7768
        T0 = 3.750*Constants.k_B
        T1 = 3.468e6*Constants.k_B
        C = 0.0223
        T2 = 7.206e4*Constants.k_B
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C, T2)

    def rr_n4(self, T):
        A = 1.553e-10
        B = 0.6682
        T0 = 1.823e2*Constants.k_B
        T1 = 7.751e6*Constants.k_B
        C = 0.0
        T2 = 0.0
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C, T2)
    
    def rr_n5(self, T):
        A = 6.245e-11
        B = 0.4985
        T0 = 1.957e3*Constants.k_B
        T1 = 2.177e7*Constants.k_B
        C = 0.0
        T2 = 0.0
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C, T2)
    
    def rr_n6(self, T):
        A = 2.388e-10
        B = 0.6732
        T0 = 3.960e2*Constants.k_B
        T1 = 3.583e7*Constants.k_B
        C = 0.0
        T2 = 0.0
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C, T2)
    
    def rr_n7(self, T):
        A = 6.170e-10
        B = 0.7481
        T0 = 1.316e2*Constants.k_B
        T1 = 3.427e7*Constants.k_B
        C = 0.0
        T2 = 0.0
    
        return self.rrxs_fitequation(A, B, T0, T1, T, C, T2)

    def ibrem_xs(self, mesh, index, nu, method=2):
        # Calculate the plasma frequency [ns]
        plasma_freq_sq = Constants.e**2*mesh.ne[index]/(Constants.me_kg*Constants.eps0)
        Z_bar = mesh.ne[index]/mesh.atom_density[index]
        
        if method == 0:
            # Drake (2016)
            coul_log = max(1, (24 - np.log(np.sqrt(mesh.ne[index])/((mesh.Te[index]*1e3)**1.5))))
            nu_ei = 3e-15*mesh.atom_density[index]*Z_bar**2/((mesh.Te[index]*1e3)**1.5)*coul_log

            sigma_ib = nu_ei/(2*Constants.c)*plasma_freq_sq/(2*np.pi*nu)**2
        elif method == 1:
            # Johnston and Dawson (1973) [Salzmann 1998 pg 221]
            vT = np.sqrt(3*mesh.Te[index]*Constants.keV2J/(Constants.me_kg))*Constants.mps2cmpns  
            classical_dist = self.Z*Constants.e**2/(mesh.Te[index]*Constants.mps2cmpns**2*Constants.keV2J*4*np.pi*Constants.eps0)
            deBroglie_wvln = Constants.h*Constants.keV2J*Constants.mps2cmpns/(2*np.pi*np.sqrt(Constants.me_kg*mesh.Te[index]*Constants.keV2J))
            pmin = max(classical_dist, deBroglie_wvln)

            lam = min(vT/(pmin*np.sqrt(plasma_freq_sq)), vT/(pmin*2*np.pi*nu))

            factor1 = 64*np.pi**3*Z_bar**2*mesh.ne[index]*Constants.e**6*np.log(lam)
            factor2 = (3*Constants.c*((2*np.pi*nu)**2 - plasma_freq_sq))*(2*np.pi*Constants.me_kg*mesh.Te[index]*Constants.keV2J*Constants.mps2cmpns**2)**1.5

            if (factor1 < 0):
                sigma_ib = 1
            elif (2*np.pi*nu)**2 > plasma_freq_sq:
                sigma_ib = factor1/(factor2*(4*np.pi*Constants.eps0)**3)
            else:
                sigma_ib = 1e6
        else:
            # Zeldovitch and Raizer (1966) [Salzmann 1998 pg 220]
            factor1 = 8*np.pi/(3*np.sqrt(3.0))*Constants.e**6*(Constants.h*Constants.c)**2/(Constants.me_keV)
            factor2 = 1/np.sqrt(2*np.pi*Constants.me_keV*mesh.Te[index])
            factor3 = (Z_bar)**2*mesh.ne[index]*(Constants.h*nu*Constants.keV2J*Constants.mps2cmpns**2)**-3

            sigma_ib = factor1*factor2*factor3/(4*np.pi*Constants.eps0)**3

        return sigma_ib

    def sigma_fitequation(self, dE, P, A, X, k, Te):
        U = dE/Te 

        is2ins = 1e-9 # Conversion factor from 1/s to 1/ns

        sigma = is2ins*A*(1 + P*np.sqrt(U))/(X + U)*U**k*np.exp(-U)

        return sigma

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

#    def E_spectral(self, Te, ni):
#        h = 4.135e-9        # Planck's constant [keV-ns]
#        a = 0.01372
#        c = 30.0            # Speed of light [cm/s]
#        keV2GJ = 1.602e-25  # Conversion factor for keV to GJ
#
#        levels = len(ni)
#        ne = np.dot(np.arange(1, levels), ni[1:])
#
#        Esp = 0.0
#
#        for i in range(levels - 1):
#            Gamma = scipy.integrate.quad(lambda nu : blackbody(nu, Te)*self.pi_n(h*nu, i)*(h*nu - self.Eth[i]), self.Eth[i]/h, self.Emax[i]/h)[0]/(a*c*Te**4/(4*np.pi*keV2GJ))
#            flux = scipy.integrate.quad(lambda nu : blackbody(nu, Te)/(h*nu), self.Eth[i]/h, self.Emax[i]/h)[0]
#
#            R = self.rr_n(Te, i + 1)*(1.5*Te + self.Eth[i])
#
#            Esp += Gamma*flux*ni[i] - R*ne*ni[i + 1]
#
#        return Esp

@jit
def cv(T, rho):
    return 1.5*rho