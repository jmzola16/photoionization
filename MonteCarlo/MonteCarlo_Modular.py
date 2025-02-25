# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:07:35 2025

This script uses a Monte Carlo method to solve a PI Front problem in planar
geometry

@author: jzola2
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import copy

# Physical constants
a = 0.01372                         # Source spectrum [GJ/(cm^3*keV^4)]
c = 30                              # Speed of light [cm/s]
sigma_SB = a*c/4                    # Stefan-Boltzmann constant [GJ/(cm^2*s*keV^4)]
keV2GJ = 1.602e-25                  # Conversion factor from keV to GJ

# Problem setup
N = 1000                            # Number of particles sourced per iteration
M = 100                             # Number of cells
edges = np.linspace(0, 0.7, M + 1)  # Cell edge location [cm]
flux = np.zeros((M, ))              # Scalar flux in each cell
t_max = 3                           # Maximum time
dt = 0.001                          # Time step [ns]
Ts = 0.1                            # Source temperature [keV]
n = 2e20                            # Number density of particles in medium [particles/cm^2]
sigma_a = 0.87e-18                  # Photoionization microscopic cross-section [cm^2]
Sigma = np.ones((M, ))*n*sigma_a    # Macroscopic photoionization cross-section

emission_rate = dt*sigma_SB*Ts**4/(2.71*Ts*keV2GJ) # Photon flux [photons/(cm^2*s)]

class Particle:
    def __init__(self, x, mu, w, cell, timestep_dist):
        self.x = x
        self.mu = mu
        self.w = w
        self.cell = cell
        self.timestep_dist = timestep_dist
        
    def move(self, bounds, Sigma):
        # Move a particle along its direction vector
        path_length = (bounds[1] - self.x)/self.mu*(self.mu > 0) + (self.x - bounds[0])/self.mu*(self.mu < 0)
        if abs(path_length) < self.timestep_dist:
            # If the particle can reach the border of the cell within the current time step, increment its cell
            self.x += path_length*(abs(self.mu))
            self.timestep_dist -= abs(path_length)
            self.cell += int(np.sign(self.mu))
        else:
            # Else move the particle within the cell
            path_length = self.timestep_dist
            self.x += self.timestep_dist*self.mu
            self.timestep_dist = 0
            
        # Return the distance travelled
        return abs(path_length)
            
    def reduce_weight(self, path_length, Sigma):
        # Reduce the particle length by the path length traveled
        self.w *= np.exp(-Sigma*path_length)

def advance_particles(census, Sigma, flux, emission_rate, seed):
    # Particle source information
    source_loc = int(0)
    w0 = 0.5*emission_rate/N
    timestep_dist = c*dt
    cutoff = 1e-10*w0
    
    # Set the random number seed for generating new particles
    rng = np.random.default_rng(seed)
    
    for i in range(N):
        # Source new particles for this timestep
        mu = np.sqrt(rng.random())
        start_time = dt*rng.random()
        photon = Particle(edges[source_loc], mu, w0, source_loc, (dt - start_time)/dt*timestep_dist)
        census.append(photon)
    
    index = 0
    while index < len(census):
        p = census[index]
        # For every particle in the system
        while p.timestep_dist > 0 and p.cell < M and p.cell >= 0:
            # Move the particle until it leaves the domain or the timestep ends
            current_cell = p.cell
            s = p.move(edges[current_cell:(current_cell + 2)], Sigma[current_cell])
            
            dx = edges[current_cell + 1] - edges[current_cell]
            
            if Sigma[current_cell] < 1e-8:
                flux[current_cell] += s*p.w/(dx*dt)
            else:
                flux[current_cell] += 1/(Sigma[current_cell]*dx*dt)*p.w*(1 - np.exp(-Sigma[current_cell]*s))
            
            p.reduce_weight(s, Sigma[current_cell])
        
        # Reset the distance each particle has traveled 
        p.timestep_dist = timestep_dist
            
        if p.w < cutoff or p.cell >= M or p.cell < 0:
            # If the particle has a weight below the cutoff or leaves the domain remove it
            census.pop(index)
        else:
            # Move on to the next particle
            index += 1
                
    return census, flux

def iterative_solver(census, nb, Sigma, flux, tol, max_it, emission_rate, seed):
    # Define error metrics and max iterations
    it = 0
    err = np.ones((2, ))*n
    
    # Copy the initial population which can be reused in each iteration
    population = copy.deepcopy(census)
    flux_old = flux.copy()
    nb_old = nb.copy()
    nb_previt = nb.copy()
    while (max(abs(err)) > tol and it < max_it):
        census = copy.deepcopy(population)
        census, flux = advance_particles(census, Sigma, 0*flux, emission_rate, seed)
        
        # Calculate the update to ionization level and associated error
        dnb = sigma_a*(n - nb)*flux*dt
        nb = nb_old + dnb
        
        # print("It:", it)
        # print("Nb old:", nb_old)
        # print("Dnb:", dnb)
        # print("Flux:", flux)
        
        err[0] = np.linalg.norm(nb - nb_previt)
        nb_previt = nb.copy()
        
        # Update average macroscopic cross section
        Sigma = sigma_a*(n - nb)
        
        # Calculate the error from flux
        err[1] = np.linalg.norm(flux - flux_old)
        flux_old[:] = flux[:]
        it += 1
        
    return census, nb, Sigma, flux

census=[]
nb = np.zeros((M, ))
for t in range(int(t_max/dt) + 1):
    #print("Time step: ", t)
    census, nb, Sigma, flux = iterative_solver(census, nb, Sigma, flux, 1e-8*n, 100, emission_rate, t)
    
    if t == 100:
        plt.subplot(2, 1, 2)
        plt.plot(edges[:-1] + np.diff(edges)/2, nb/n, label='t=0.1')
        plt.subplot(2, 1, 1)
        plt.plot(edges[:-1] + np.diff(edges)/2, flux, label='t=0.1')
    elif t == 450:
        plt.subplot(2, 1, 2)
        plt.plot(edges[:-1] + np.diff(edges)/2, nb/n, label='t=0.45')
        plt.subplot(2, 1, 1)
        plt.plot(edges[:-1] + np.diff(edges)/2, flux, label='t=0.45')
    elif t == 750:
        plt.subplot(2, 1, 2)
        plt.plot(edges[:-1] + np.diff(edges)/2, nb/n, label='t=0.75')
        plt.subplot(2, 1, 1)
        plt.plot(edges[:-1] + np.diff(edges)/2, flux, label='t=0.75')
    elif t == 1000:
        plt.subplot(2, 1, 2)
        plt.plot(edges[:-1] + np.diff(edges)/2, nb/n, label='t=1')
        plt.subplot(2, 1, 1)
        plt.plot(edges[:-1] + np.diff(edges)/2, flux, label='t=1')
    elif t == 2000:
        plt.subplot(2, 1, 2)
        plt.plot(edges[:-1] + np.diff(edges)/2, nb/n, label='t=2')
        plt.subplot(2, 1, 1)
        plt.plot(edges[:-1] + np.diff(edges)/2, flux, label='t=2')

# for t in range(int(t_max/dt) + 1):
#     census, flux = advance_particles(census, Sigma, flux*0, emission_rate, t)

# ana = scipy.special.expn(2, Sigma[0]*(edges[:-1] + np.diff(edges)/2))
plt.subplot(2, 1, 2)
plt.plot(edges[:-1] + np.diff(edges)/2, nb/n, label='t='+str(t_max))
plt.xlabel('x-location [cm]')
plt.ylabel('Ionization fraction')
plt.subplot(2, 1, 1)
plt.plot(edges[:-1] + np.diff(edges)/2, flux, label='t='+str(t_max))
# plt.plot(edges[:-1] + np.diff(edges)/2, ana, label='Analytical Solution')
plt.ylabel('Scalar Flux in cell [photons/(cm^2*s)]')
plt.legend()
plt.show()