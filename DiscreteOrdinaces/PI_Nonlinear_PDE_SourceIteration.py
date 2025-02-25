# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 08:33:20 2024

This program solves for the scalar flux and ionization fraction in a photoionization
front using an Sn method with source iteration

@author: James Zola
"""

import numpy as np
import matplotlib.pyplot as plt
from PI_Curvature import front_curvature_paper_data  

N = 8
M = 100
R = 0.1 # cm
L = 0.7 # cm
c = 30 # cm/ns
dt = 0.001 # ns
t_max = 3
dz = L/M
z = np.linspace(0, L, M)
n = 2e20 #1
kappa = 0.87e-18 # cm^2
sigma_s = n*5.1e-27 # cm^2
T = 0.100 # keV
f = (np.ones((N, ))*0.01372*c*T**3)/(4*2.71*1.602e-25)
a = 0 #2/(R*(np.pi))

mu, w = np.polynomial.legendre.leggauss(N)

nb_old = np.zeros((M, ))
nb_guess = np.zeros((M, ))
nb_guess[0] = 1
nb = np.zeros((M, ))
temp = np.zeros((M, ))

psi_old = np.zeros((M, N))
psi = np.zeros((M, N))
phi = np.matmul(psi, w)
phi_old = np.matmul(psi, w)
phi_it = np.matmul(psi, w)

plt.subplots(2, 1)

for t in range(int(t_max/dt) + 1):
    err1 = n
    err2 = n
    tol = n*1e-8
    it = 0
    while((err1 > tol or err2 > tol) and it < 50):
        psi[0, int(N/2):] = (1/(c*dt)*psi_old[0, int(N/2):] + mu[int(N/2):]*f[int(N/2):]/(dz) + sigma_s/2*phi_it[0])/(1/(c*dt) + mu[int(N/2):]/(dz) + (1 - mu[int(N/2):]**2)**(0.5)*a + (n - nb_guess[0])*kappa + sigma_s) # (n - nb_old[0])*kappa/(1 + kappa*dt*phi_old[0])
        psi[-1, :int(N/2)] = (1/(c*dt)*psi_old[-1, :int(N/2)] + sigma_s/2*phi_it[-1])/(1/(c*dt) - mu[:int(N/2)]/(dz) + (1 - mu[:int(N/2)]**2)**(0.5)*a + (n - nb_guess[-1])*kappa + sigma_s) # (n - nb_old[-1])*kappa/(1 + kappa*dt*phi_old[-1])
        for j in range(1, M):
            psi[j, int(N/2):] = (1/(c*dt)*psi_old[j, int(N/2):] + mu[int(N/2):]*psi[j - 1, int(N/2):]/(dz) + sigma_s/2*phi_it[j])/(1/(c*dt) + mu[int(N/2):]/(dz) + (1 - mu[int(N/2):]**2)**(0.5)*a + (n - nb_guess[j])*kappa + sigma_s) # (n - nb_old[j])*kappa/(1 + kappa*dt*phi_old[j])
            psi[M - j - 1, :int(N/2)] = (1/(c*dt)*psi_old[M - j, :int(N/2)] - mu[:int(N/2)]*psi[M - j, :int(N/2)]/(dz) + sigma_s/2*phi_it[M - j])/(1/(c*dt) - mu[:int(N/2)]/(dz) + (1 - mu[:int(N/2)]**2)**(0.5)*a + (n - nb_guess[M - j])*kappa + sigma_s) # (n - nb_old[j])*kappa/(1 + kappa*dt*phi_old[M - j])
            
        phi_it = np.matmul(psi, w)
        
        nb = ((1/dt)*nb_old + n*kappa*phi_it)/(1/dt + kappa*phi_it)
        
        err1 = np.linalg.norm(phi - phi_it)
        err2 = np.linalg.norm(nb - nb_guess)
        
        nb_guess[:] = nb[:]
        
        phi[:] = phi_it[:]
        
        it += 1
    
     
    #temp[:] = nb[:]
    #nb = nb_old + dt*kappa*(n - nb_old)*phi/(1 + kappa*dt*phi_old)
    #nb_old[:] = temp[:]

    psi_old[:] = psi[:]
    psi = np.zeros((M, N))
    phi_old[:] = phi[:]
    
    nb_old = nb
    
    # if (dt*t > p/c and p < 7):
    #     plt.figure(1)
    #     plt.plot(z, nb/n, label='t=' + str(dt*t))
    #     plt.figure(2)
    #     plt.plot(z, phi, label='t=' + str(dt*t))

    #     p += 1
    
    if t == 100:
        #plt.figure(1)
        plt.subplot(2, 1, 2)
        plt.plot(z, nb/n, label='t=0.1')
        #plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(z, phi, label='t=0.1')
    elif t == 450:
        #plt.figure(1)
        plt.subplot(2, 1, 2)
        plt.plot(z, nb/n, label='t=0.45')
        #plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(z, phi, label='t=0.45')
    elif t == 750:
        #plt.figure(1)
        plt.subplot(2, 1, 2)
        plt.plot(z, nb/n, label='t=0.75')
        #plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(z, phi, label='t=0.75')
    elif t == 1000:
        #plt.figure(1)
        plt.subplot(2, 1, 2)
        plt.plot(z, nb/n, label='t=1')
        #plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(z, phi, label='t=1')
    elif t == 2000:
        #plt.figure(1)
        plt.subplot(2, 1, 2)
        plt.plot(z, nb/n, label='t=2')
        #plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(z, phi, label='t=2')

        
#plt.figure(1)
plt.subplot(2, 1, 2)
plt.plot(z, nb/n, label='t='+str(t_max))   
plt.legend()
plt.xlabel('z-location [cm]')
plt.ylabel('Fraction of $n_B$')

#plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(z, phi, label='t='+str(t_max))
plt.title('Constant Basis Galerkin Model') 
plt.legend()
plt.ylabel('Scalar Flux, $\phi$')
plt.show()