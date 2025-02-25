# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:54:59 2024

This program models a photoionization front propagating through a circular duct
using a weighted residual method to represent the radial dependence of the flux
using basis functions and a discrete ordinance method to solve the differential
equation. 

@author: jzola2
"""
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define parameters
N = 8       # Number of discrete ordinances
M = 100     # Number of spatial cells in L
P = 2       # Number of basis functions
M2 = 100    # Number of spatial cells in R

L = 0.7         # Duct length [cm]
R = 0.1         # Duct radius [cm]
C = 2*np.pi*R   # Duct circumference [cm]
c = 30          # Speed of light [cm/ns]
dt = 0.001      # Time step [ns]
t_max = 3       # End time [ns]
dz = L/M        # Cell width [cm]

z = np.linspace(0, L, M)    # Cell edges (length) [cm]
rho = np.linspace(0, R, M2) # Cell edges (radius) [cm]
n = 2e20                    # Particle density [particles/cm^3]
kappa = 0.87e-18            # Photoionization cross section [cm^2]
sigma_s = n*5.1e-27         # Scattering cross section [cm^2]
Ts = 0.100                  # Source temperature [eV]

IC = 0                      # IC (0 for constant, 1 for Gaussian)
Diss = 0                    # Use high dissipation for integrals (1 for True, 0 for False)

mu, w = np.polynomial.legendre.leggauss(N)

# Create dependent variables
psi = np.zeros((M, P, N))
psi_old = np.zeros((M, P, N))

Phi = np.zeros((M, P))
Phi_guess = np.zeros((M, P))

nb = np.zeros((M, P))
nb_old = np.zeros((M, P))
nb_guess = np.zeros((M, P))
nb_guess[0, 0] = 1


# Define basis functions and associated constants

# @njit
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

# def alpha_squared(r, phi):
#     return r*alpha(r, phi)**2

# def alpha_cubed(r, phi):
#     return r*alpha(r, phi)**3

epsilon = np.zeros((P, P, P))

if Diss == 0:
    gamma123 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 1)*alpha(r, phi, 2)*alpha(r, phi, 3), 0, 2*np.pi, 0, R)[0]
    gamma222 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 2)**3, 0, 2*np.pi, 0, R)[0]
    gamma333 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 3)**3, 0, 2*np.pi, 0, R)[0]
    gamma223 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 3)*alpha(r, phi, 2)**2, 0, 2*np.pi, 0, R)[0]
    gamma233 = 1/(np.pi*R**2)*integrate.dblquad(lambda r, phi : r*alpha(r, phi, 2)*alpha(r, phi, 3)**2, 0, 2*np.pi, 0, R)[0]
else:
    gamma123 = 1/(np.pi*R**2)*(R/2)*alpha(R/2, np.pi, 1)*alpha(R/2, np.pi, 2)*alpha(R/2, np.pi, 3)*R*2*np.pi
    gamma222 = 1/(np.pi*R**2)*(R/2)*alpha(R/2, np.pi, 2)**3*(R)*2*np.pi
    gamma333 = 1/(np.pi*R**2)*(R/2)*alpha(R/2, np.pi, 3)**3*R*2*np.pi
    gamma223 = 1/(np.pi*R**2)*(R/2)*alpha(R/2, np.pi, 3)*alpha(R/2, np.pi, 2)**2*R*2*np.pi
    gamma233 = 1/(np.pi*R**2)*(R/2)*alpha(R/2, np.pi, 2)*alpha(R/2, np.pi, 3)**2*R*2*np.pi
# gamma2 = 1/(np.pi*R**2)*integrate.dblquad(alpha_squared, 0, 2*np.pi, 0, R)[0]
# gamma3 = 1/(np.pi*R**2)*integrate.dblquad(alpha_cubed, 0, 2*np.pi, 0, R)[0]

for p in range(P):
    for q in range(P):
        for s in range(P):
            if (p == q and q == s and p == 0):
                epsilon[p, q, s] = 1
            if (p != q and p != s and q != s):
                epsilon[p, q, s] = 0 #gamma123
            if (p == q and q == s and p == 1):
                epsilon[p, q, s] = gamma222
            if (p == q and q == s and p == 2):
                epsilon[p, q, s] = gamma333
            if ((p == 1 and q == 1 and s != 1) or (q == 1 and s == 1 and p != 1) or (p == 1 and s == 1 and q != 1)):
                epsilon[p, q, s] = gamma223
            if ((p == 2 and q == 2 and s != 2) or (q == 2 and s == 2 and p != 2) or (p == 2 and s == 2 and q != 2)):
                epsilon[p, q, s] = gamma233
            if ((p == 0 and q == s) or (q == 0 and p == s) or (s == 0 and p == q)):
                epsilon[p, q, s] = 1

# Define initial condition
a = 0.01372
f = np.zeros((P, N))
peak = (a*c*Ts**3/(4*2.71*1.602e-25))
if IC == 0:
    f[0, :] = np.ones((N, ))*peak
if IC == 1:
    std_dev = R/3
    approx = np.zeros((M2, ))
    for i in range(P):
        const = (1/(np.pi*R**2))*integrate.dblquad(lambda r, phi : r/(std_dev*(2*np.pi)**(0.5))*np.exp(-r**2/(2*std_dev**2))*alpha(r, phi, i + 1), 0, 2*np.pi, 0, R)[0]
        f[i, :] = const*np.ones((N, ))*peak
        for j in range(M2):
            approx[j] += np.sum(f[i, :]*w)*(1/(2*np.pi))*integrate.quad(lambda phi : alpha(rho[j], phi, i + 1), 0, 2*np.pi)[0]
    
plt.plot(rho, approx, label='Approximate Source')
plt.plot(rho, (peak)/(std_dev*(2*np.pi)**(0.5))*np.exp(-rho**2/(2*std_dev**2)), label='Gaussian Source')
plt.plot(rho, peak*np.ones(rho.size))
plt.xlabel('r location [cm]')
plt.ylabel('Incoming Flux')
plt.legend()
plt.show()
assert 0

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
for t in range(int(t_max/dt)):
    # Define error metrics for convergence
    it = 0
    err = np.ones((2*P, ))*n
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
                        Q[k, l] = -kappa*np.sum(epsilon[k, l, :]*nb_guess[0, :]) + A[k, l]*(1 - mu[j]**2)**0.5
                    else:
                        # Contributions from other nodes (off diagonal)
                        Q[k, l] = 1/(c*dt) + abs(mu[j])/dz + kappa*n - kappa*np.sum(epsilon[k, l, :]*nb_guess[0, :]) + (1 - mu[j]**2)**(0.5)*A[k, l] + sigma_s
            
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
                            Q[k, l] = -kappa*np.sum(epsilon[k, l, :]*nb_guess[i, :]) + A[k, l]*(1 - mu[j]**2)**0.5
                        else:
                            # Node being solved for (on diagonal)
                            Q[k, l] = 1/(c*dt) + abs(mu[j])/dz + kappa*n - kappa*np.sum(epsilon[k, l, :]*nb_guess[i, :]) + (1 - mu[j]**2)**(0.5)*A[k, l] + sigma_s
                
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
            for k in range(P):
                for l in range(P):
                    if k != l:
                        # Contributions from other nodes (off diagonal)
                        Q[k, l] = kappa*(np.sum(epsilon[k, :, l]*Phi[i, :]))
                    else:
                        # Node being solved for (on diagonal)
                        Q[k, l] = 1/dt + kappa*(np.sum(epsilon[k, :, l]*Phi[i, :]))
            
            # Known quantities
            rhs = nb_old[i, :]/dt + kappa*n*Phi[i, :]
            
            nb[i, :] = np.linalg.solve(Q, rhs)
         
        # Update error by comparing new solutions to guesses
        for p in range(P):
            err[p] = np.linalg.norm(nb[:, p] - nb_guess[:, p])
            err[P + p] = np.linalg.norm(Phi[:, p] - Phi_guess[:, p])
        
        # Guess new solution to check convergence
        nb_guess[:] = nb[:]
        Phi_guess[:] = Phi[:]
            
        it += 1
    
    # Update variables
    nb_old[:] = nb[:]
    psi_old[:] = psi[:]
    
    # Plot at relevant times
    if t == 100 or t == 450 or t == 750 or t == 1000 or t == 2000:
        flux = np.zeros((M2, M))
        ions = np.zeros((M2, M))
        for edge in range(M2):
            for i in range(P):
                beta = 1/(2*np.pi)*integrate.quad(lambda phi : alpha(rho[edge], phi, i + 1), 0, 2*np.pi)[0]
                flux[edge, :] += Phi[:, i]*beta
                ions[edge, :] += nb[:, i]*beta
            #flux[edge, :] = Phi[:, 0] + Phi[:, 1]*beta #alpha(rho[edge], 0)
            #ions[edge, :] = (nb[:, 0] + nb[:, 1])/n*beta #alpha(rho[edge], 0))/n
        ions = ions/n
        
        # for i in range(M2):
        #     for j in range(M):
        #         if ions[i, j] < 0:
        #             ions[i, j] = 0
            
        fig, ax = plt.subplots(2, 1)
        ax = plt.subplot(2, 1, 1)
        im = ax.pcolormesh(z, rho, flux)
        ax.axis('equal')
        plt.ylabel('Radius [cm]')
        plt.title('Scalar flux at t=' + str(t*dt))
        fig.colorbar(im, ax=ax)
        
        ax = plt.subplot(2, 1, 2)
        im = ax.pcolormesh(z, rho, ions)
        ax.axis('equal')
        plt.xlabel('Distance [cm]')
        plt.ylabel('Radius [cm]')
        plt.title('Ionization fraction at t=' + str(t*dt))
        fig.colorbar(im, ax = ax)
        plt.show()

# Plot the final state
flux = np.zeros((M2, M))
ions = np.zeros((M2, M))
for edge in range(M2):
    for i in range(P):
        beta = 1/(2*np.pi)*integrate.quad(lambda phi : alpha(rho[edge], phi, i + 1), 0, 2*np.pi)[0]
        flux[edge, :] += Phi[:, i]*beta
        ions[edge, :] += nb[:, i]*beta
    #flux[edge, :] = Phi[:, 0] + Phi[:, 1]*beta #alpha(rho[edge], 0)
    #ions[edge, :] = (nb[:, 0] + nb[:, 1])/n*beta #alpha(rho[edge], 0)

ions = ions/n

# for i in range(M2):
#     for j in range(M):
#         if ions[i, j] < 0:
#             ions[i, j] = 0
        
fig, ax = plt.subplots(2, 1)
ax = plt.subplot(2, 1, 1)
im = ax.pcolormesh(z, rho, flux)
ax.axis('equal')
plt.ylabel('Radius [cm]')
plt.title('Scalar flux at t=' + str(t*dt))
fig.colorbar(im, ax=ax)

ax = plt.subplot(2, 1, 2)
im = ax.pcolormesh(z, rho, ions)
ax.axis('equal')
plt.xlabel('Distance [cm]')
plt.ylabel('Radius [cm]')
plt.title('Ionization fraction at t=' + str(t*dt))
fig.colorbar(im, ax = ax)
plt.show()