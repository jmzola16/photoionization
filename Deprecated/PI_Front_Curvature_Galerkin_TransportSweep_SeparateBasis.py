# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:54:59 2024

This program models a photoionization front propagating through a circular duct
using a weighted residual method to represent the radial dependence of the flux
using basis functions and a discrete ordinance method to solve the differential
equation. 

@author: jzola2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define parameters
N = 8       # Number of discrete ordinances
M = 100     # Number of spatial cells in L
P = 2       # Number of flux basis functions
S = 4       # Number of ionization basis functions

L = 0.7         # Duct length [cm]
R = 0.1         # Duct radius [cm]
C = 2*np.pi*R   # Duct circumference [cm]
c = 30          # Speed of light [cm/ns]
dt = 0.001      # Time step [ns]
t_max = 3       # End time [ns]
dz = L/M        # Cell width [cm]
M2 = 100        # Number of spatial cells in R

z = np.linspace(0, L, M)    # Cell edges (length) [cm]
rho = np.linspace(0, R, M2) # Cell edges (radius) [cm]
kappa = 0.87e-18            # Photoionization cross section [cm^2]
n = 2e20                    # Particle density [particles/cm^3]
sigma_s = n*5.1e-27         # Scattering cross section [cm^2]
Ts = 0.100                  # Source temperature [eV]

IC = 1                      # IC (0 for constant, 1 for Gaussian)

mu, w = np.polynomial.legendre.leggauss(N)

# Create dependent variables
psi = np.zeros((M, P, N))
psi_old = np.zeros((M, P, N))

Phi = np.zeros((M, P))
Phi_guess = np.zeros((M, P))

nb = np.zeros((M, S))
nb_old = np.zeros((M, S))
nb_guess = np.zeros((M, S))
nb_guess[0, 0] = 1

# Limiting polynomial reconstructions
M0 = n
m0 = 0

# Define basis functions and associated constants
def alpha(r, phi, basis):
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
    
def beta(r, basis):
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

def beta_bar(basis):
    if basis == 1:
        #return np.sqrt(2)/2
        return np.sqrt(2)/R
    elif basis == 2:
        #return 0
        return 2*np.sqrt(6/R**6)*(R**2/3 - R**2/2)
    elif basis == 3:
        #return 0
        return np.sqrt(10/R**10)*(6/5*R**4 - 2*R**4 + R**4)
    elif basis == 4:
        #return 0
        return np.sqrt(14/R**14)*(-20/7*R**6 + 6*R**6 - 4*R**6 + R**6)
    elif basis == 5:
        #return 0
        return np.sqrt(18/R**18)*(70/9*R**8 - 20*R**8 + 18*R**8 - 20/3*R**8 + R**8)

# def alpha_squared(r, phi):
#     return r*alpha(r, phi)**2

# def alpha_cubed(r, phi):
#     return r*alpha(r, phi)**3

epsilon = np.zeros((S, P, S))
iota = np.zeros((S, P))
gamma = np.zeros((P, P, S))

#gamma123 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 1)*alpha(r, phi, 2)*alpha(r, phi, 3), 0, 2*np.pi, 0, R)[0]
#gamma222 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 2)**3, 0, 2*np.pi, 0, R)[0]
#gamma333 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 3)**3, 0, 2*np.pi, 0, R)[0]
#gamma223 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 3)*alpha(r, phi, 2)**2, 0, 2*np.pi, 0, R)[0]
#gamma233 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 2)*alpha(r, phi, 3)**2, 0, 2*np.pi, 0, R)[0]
# gamma2 = 1/(np.pi*R**2)*integrate.dblquad(alpha_squared, 0, 2*np.pi, 0, R)[0]
# gamma3 = 1/(np.pi*R**2)*integrate.dblquad(alpha_cubed, 0, 2*np.pi, 0, R)[0]

for p in range(S):
    for q in range(P):
        for s in range(S):
            epsilon[p, q, s] = 1/(2*np.pi)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, q + 1)*beta(r, p + 1)*beta(r, s + 1), 0, 2*np.pi, 0, R)[0]
            if s == 0:
                iota[p, q] = 1/(2*np.pi)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, q + 1)*beta(r, p + 1), 0, 2*np.pi, 0, R)[0]
                
for p in range(P):
    for q in range(P):
        for s in range(S):
            gamma[p, q, s] = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, p + 1)*alpha(r, phi, q + 1)*beta(r, s + 1), 0, 2*np.pi, 0, R)[0]

# Define initial condition
a = 0.01372
f = np.zeros((P, N))
peak = (a*c*Ts**3/(2.71*1.602e-25))
if IC == 0:
    f[0, :] = np.ones((N, ))*peak
if IC == 1:
    std_dev = R/3
    #approx = np.zeros((M2, ))
    for i in range(P):
        const = (1/(np.pi*R**2))*integrate.dblquad(lambda r, phi : r/(std_dev*(2*np.pi)**(0.5))*np.exp(-r**2/(2*std_dev**2))*alpha(r, phi, i + 1), 0, 2*np.pi, 0, R)[0]
        f[i, :] = const*np.ones((N, ))*peak
        # for j in range(M2):
        #     approx[j] += np.sum(f[i, :]*w)*(1/(2*np.pi))*integrate.quad(lambda phi : alpha(rho[j], phi, i + 1), 0, 2*np.pi)[0]
    
# plt.plot(rho, approx)
# plt.plot(rho, (a*c*Ts**3/(2.71*1.602*10e-25))/(std_dev*(2*np.pi)**(0.5))*np.exp(-rho**2/(2*std_dev**2)))
# plt.show()


# Define geometric absorption matrix
A = np.zeros((P, P))
A[0, 0] = 2/(np.pi*R)
if P > 1:
    A[0, 1] = ((3*np.pi - (16/np.pi))*(9*np.pi**2 - 64)**(-0.5))/R
    A[1, 0] = (-16*(9*np.pi**2 - 64)**(-0.5))/(np.pi*R)
    A[1, 1] = (128*(9*np.pi**2 - 64)**(-1))/(np.pi*R)
if P > 2:
    u = 3*np.pi*(9*np.pi**2 - 64)**(-0.5)/R
    v = 8*R/(3*np.pi)
    
    q = 8*R*(9*np.pi/5*(9*np.pi**2 - 64)**(-1) - 2/(3*np.pi))
    p = R**(-2)*(1 - 576/25*(9*np.pi**2 - 64)**(-1))**(-0.5)
    
    A[0, 2] = -q*p + (v**2 + q*v - 1/(u**2))*p*C/(np.pi**2*R**2)
    A[1, 2] = (2*p/u) - (v**2 + q*v - 1/(u**2))*u*v*p*C/(np.pi**2*R**2)
    A[2, 0] = (v**2 + q*v - 1/(u**2))*p*C/(np.pi**2*R**2)
    A[2, 1] = -(v**2 + q*v - 1/(u**2))*u*v*p*C/(np.pi**2*R**2)
    A[2, 2] = (v**2 + q*v - 1/(u**2))**2*p**2*C/(np.pi**2*R**2)

# Loop over time
for t in range(int(t_max/dt) + 1):
    # Define error metrics for convergence
    it = 0
    err = np.ones((P + S, ))*n
    tol = n*1e-8
    while (max(err) > tol and it < 50):
        # Loop over angles
        for j in range(N):
            # Define matrix to solve for coupled terms in each angle
            Q = np.zeros((P, P))
            # Loop over each WR node
            for k in range(P):
                for l in range(P):
                    if k != l:
                        # Node being solved for (on diagonal)
                        Q[k, l] = -kappa*np.sum(gamma[k, l, :]*nb_guess[0, :]) + A[k, l]*(1 - mu[j]**2)**0.5
                    else:
                        # Contributions from other nodes (off diagonal)
                        Q[k, l] = 1/(c*dt) + abs(mu[j])/dz + kappa*n - kappa*np.sum(gamma[k, l, :]*nb_guess[0, :]) + (1 - mu[j]**2)**(0.5)*A[k, l] + sigma_s
            
            # Knowns for each equation
            rhs = sigma_s/2*Phi_guess[0, :] + 1/(c*dt)*psi_old[0, :, j] + mu[j]/dz*f[:, j]*(mu[j] > 0)
            
            psi[0, :, j] = np.linalg.solve(Q, rhs)
                        
        # Loop over all spatial cells
        for i in range(1, M):
            # Loop over all angles
            for j in range(N):
                # Set up system of equations
                Q = np.zeros((P, P))
                # Loop over WR nodes
                for k in range(P):
                    for l in range(P):
                        if k != l:
                            # Contributions from other nodes (off diagonal)
                            Q[k, l] = -kappa*np.sum(gamma[k, l, :]*nb_guess[i, :]) + A[k, l]*(1 - mu[j]**2)**0.5
                        else:
                            # Node being solved for (on diagonal)
                            Q[k, l] = 1/(c*dt) + abs(mu[j])/dz + kappa*n - kappa*np.sum(gamma[k, l, :]*nb_guess[i, :]) + (1 - mu[j]**2)**(0.5)*A[k, l] + sigma_s
                
                # Knowns for each equation (depends on mu, because flux could be left going or right going)
                if (mu[j] > 0):
                    rhs = sigma_s/2*Phi_guess[i, :] + 1/(c*dt)*psi_old[i, :, j] + mu[j]/dz*psi[i - 1, :, j]
                    psi[i, :, j] = np.linalg.solve(Q, rhs)
                else:
                    rhs = sigma_s/2*Phi_guess[M - i - 1, :] + 1/(c*dt)*psi_old[M - i - 1, :, j] - mu[j]/dz*psi[M - i, :, j]
                    psi[M - i - 1, :, j] = np.linalg.solve(Q, rhs)
                    
        for k in range(P):
            # Compute scalar flux at each node
            Phi[:, k] = np.matmul(psi[:, k, :], w)
        
        # Loop over all spatial cells
        for i in range(M):
            # Loop over WR nodes
            T = np.zeros((S, S))
            for k in range(S):
                for l in range(S):
                    if k != l:
                        # Contributions from other nodes (off diagonal)
                        T[k, l] = kappa*(np.sum(epsilon[k, :, l]*Phi[i, :]))
                    else:
                        # Node being solved for (on diagonal)
                        T[k, l] = 1/dt + kappa*(np.sum(epsilon[k, :, l]*Phi[i, :]))
            
            # Known quantities
            rhs = nb_old[i, :]/dt + kappa*n*np.matmul(iota, Phi[i, :])
            
            nb[i, :] = np.linalg.solve(T, rhs)
        
        # Limiting polynomial reconstruction
        # p_bar = np.zeros((M, ))
        # theta = np.zeros((M, ))
        # for i in range(S):
        #     p_bar += beta_bar(i + 1)*nb[:, i]
        # for j in range(M):
        #     poly = np.zeros((M2, ))
        #     for i in range(S):
        #         poly += nb[j, i]*beta(rho, i + 1)
        #     M_prime = max(poly)
        #     m_prime = min(poly)
        #     theta[j] = min(abs((M0 - p_bar[j])/(M_prime - p_bar[j])), abs((m0 - p_bar[j])/(m_prime - p_bar[j])), 1)
        
        # nb[:, 0] = (nb[:, 0] - p_bar)*theta + p_bar
        # for i in range(S - 1):
        #     nb[:, i + 1] = theta*nb[:, i + 1] #(nb[:, i] - p_bar)*theta + p_bar
            
        # if any(theta < 1):
        #     for i in range(M):
        #         if theta[i] < 1:
        #             temp_plot = np.zeros((M2, ))
        #             for j in range(S):
        #                 temp_plot += nb[i, j]*beta(rho, j + 1)
                        
        #             plt.plot(rho, temp_plot)
        #             plt.show()
        
        # Update error by comparing new solutions to guesses
        for s in range(S):
            err[s] = np.linalg.norm(nb[:, s] - nb_guess[:, s])
        for p in range(P):
            err[S + p] = np.linalg.norm(Phi[:, p] - Phi_guess[:, p])
        
        # Guess new solution to check convergence
        nb_guess[:] = nb[:]
        Phi_guess[:] = Phi[:]
            
        it += 1
    
    nb_old[:] = nb[:]
    psi_old[:] = psi[:]
    
    # Plot at relevant times
    if t == 100 or t == 450 or t == 750 or t == 1000 or t == 2000:
        flux = np.zeros((M2, M))
        ions = np.zeros((M2, M))
        for edge in range(M2):
            for i in range(P):
                alpha_phi = 1/(2*np.pi)*integrate.quad(lambda phi : alpha(rho[edge], phi, i + 1), 0, 2*np.pi)[0]
                flux[edge, :] += Phi[:, i]*alpha_phi
                
            for i in range(S):
                ions[edge, :] += nb[:, i]*beta(rho[edge], i + 1)
            #flux[edge, :] = Phi[:, 0] + Phi[:, 1]*beta #alpha(rho[edge], 0)
            #ions[edge, :] = (nb[:, 0] + nb[:, 1])/n*beta #alpha(rho[edge], 0))/n
        ions = ions/n
        
        # for i in range(M2):
        #     for j in range(M):
        #         if ions[i, j] < 0:
        #             ions[i, j] = 0
            
        fig, axs = plt.subplots(2, 1)
        #ax = plt.subplot(2, 1, 1)
        im = axs[0].pcolormesh(z, rho, flux)
        axs[0].axis('equal')
        axs[0].set_ylabel('Radius [cm]')
        axs[0].set_title('Scalar flux at t=' + str(t*dt))
        axs[0].set_xticks([])
        fig.colorbar(im, ax=axs[0])
        
        #ax = plt.subplot(2, 1, 2)
        im = axs[1].pcolormesh(z, rho, ions)
        axs[1].axis('equal')
        plt.xlabel('Distance [cm]')
        plt.ylabel('Radius [cm]')
        plt.title('Ionization fraction at t=' + str(t*dt))
        fig.colorbar(im, ax = axs[1])
        plt.show()

# Plot the final state
flux = np.zeros((M2, M))
ions = np.zeros((M2, M))
for edge in range(M2):
    for i in range(P):
        alpha_phi = 1/(2*np.pi)*integrate.quad(lambda phi : alpha(rho[edge], phi, i + 1), 0, 2*np.pi)[0]
        flux[edge, :] += Phi[:, i]*alpha_phi
    
    for i in range(S):
        ions[edge, :] += nb[:, i]*beta(rho[edge], i + 1)
    #flux[edge, :] = Phi[:, 0] + Phi[:, 1]*beta #alpha(rho[edge], 0)
    #ions[edge, :] = (nb[:, 0] + nb[:, 1])/n*beta #alpha(rho[edge], 0)

ions = ions/n

# for i in range(M2):
#     for j in range(M):
#         if ions[i, j] < 0:
#             ions[i, j] = 0
            
fig, axs = plt.subplots(2, 1)
#ax = plt.subplot(2, 1, 1)
im = axs[0].pcolormesh(z, rho, flux)
axs[0].axis('equal')
axs[0].set_ylabel('Radius [cm]')
axs[0].set_title('Scalar flux at t=' + str(t*dt))
axs[0].set_xticks([])
fig.colorbar(im, ax=axs[0])

#ax = plt.subplot(2, 1, 2)
im = axs[1].pcolormesh(z, rho, ions)
axs[1].axis('equal')
plt.xlabel('Distance [cm]')
plt.ylabel('Radius [cm]')
plt.title('Ionization fraction at t=' + str(t*dt))
fig.colorbar(im, ax = axs[1])
plt.show()