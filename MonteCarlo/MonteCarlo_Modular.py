# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:07:35 2025

This script uses a Monte Carlo method to solve a PI Front problem in planar
geometry

@author: jzola2
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
sys.path.append("..")
import CrossSectionFunctions as xsf

# Physical constants
a = 0.01372                         # Source spectrum [GJ/(cm^3*keV^4)]
c = 30                              # Speed of light [cm/ns]
sigma_SB = a*c/4                    # Stefan-Boltzmann constant [GJ/(cm^2*ns*keV^4)]
keV2GJ = 1.602e-25                  # Conversion factor from keV to GJ
h = 4.415e-9                        # Planck's constant [keV-ns]
keV2J = 1.602e-16                   # Conversion factor from keV to J
k_B = 8.617e-8                      # Boltzmann constant [keV/K]
Na = 6.6022e23                      # Avogadro's number [#/mol]
rho_N = 0.02802                     # Molar mass of Nitrogen [kg/mol]

# Problem setup
N_source = 1000                     # Number of particles sourced per iteration
N_recomb_tot = 10                   # Number of particles from recombination per cell
M = 100                             # Number of cells
edges = np.linspace(0, 0.7, M + 1)  # Cell edge location [cm]
flux = np.zeros((M, ))              # Scalar flux in each cell
T = np.ones((M, ))*1e-5             # Material temperature in each cell [keV]
t_max = 3                           # Maximum time
dt = 0.0005                         # Time step [ns]
Ts = 0.1                            # Source temperature [keV]
n = 2e20                            # Number density of particles in medium [particles/cm^3]
Gamma_nu = [xsf.pi_n1, xsf.pi_n2, xsf.pi_n3, xsf.pi_n4]  # Photoionization microscopic cross-section [cm^2]
R = [xsf.rr_n1, xsf.rr_n2, xsf.rr_n3, xsf.rr_n4]      # Radiative recombination rate constant [cm^2/s]
levels = len(Gamma_nu) + 1             # Number of ionization levels
Eth = np.array([14.53, 29.60, 47.45, 77.47])*1e-3   # Threshhold energy [keV]
N_recomb = np.zeros((M, levels - 1)) # The number of particles to source from each ionization level
# cv = 741/(keV2J*k_B)*(n/(Na)*rho_N)  # Material specific heat [keV/(keV*cm^3)]
#Sigma = np.ones((M, ))*n*sigma_a    # Macroscopic photoionization cross-section

emission_rate = dt*sigma_SB*Ts**4/(keV2GJ)  # Energy density of photons emitted [keV/(cm^2)]

class Particle:
    def __init__(self, x, mu, nu, w, cell, timestep_dist):
        self.x = x
        self.mu = mu
        self.w = w
        self.nu = nu
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

def source_particles(census, rng, dt, source_loc, w0, number, **kwargs):
    flag = 0
    for key, value in kwargs.items():
        if key == 'blackbody':
            Ts = value
            flag = 1
        elif key == 'recombination':
            Eth = value
            flag = 2
    
    if flag == 0:
        flag = 1
        Ts = 0.1
    
    for i in range(number):
        mu = np.sqrt(rng.random())
        start_time = dt*rng.random()
        
        if flag == 1:
            xi = rng.random()
            nu = xsf.sample_blackbody(Ts, xi)
        elif flag == 2:
            nu = Eth/h
            
        timestep_dist = c*dt
        
        photon = Particle(edges[source_loc], mu, nu, w0/(h*nu), source_loc, (dt - start_time)/dt*timestep_dist)
        census.append(photon)
        
    return census

def advance_particles(census, ni, T, emission_rate, seed, *varargs):
    if len(varargs) > 0:
        ni_old = varargs[0]
    else:
        ni_old = ni
    
    # Boundary source information
    source_loc = int(0)
    w0 = 0.5*emission_rate/N_source
    timestep_dist = c*dt
    cutoff = 1e-10*w0
    
    # Set the random number seed for generating new particles
    rng = np.random.default_rng(seed)
    
    for i in range(N_source):
        # Source new particles for this timestep
        mu = np.sqrt(rng.random())
        start_time = dt*rng.random()
        xi = rng.random()
        
        nu = xsf.sample_blackbody(Ts, xi)
        photon = Particle(edges[source_loc], mu, nu, w0/(h*nu), source_loc, (dt - start_time)/dt*timestep_dist)
        census.append(photon)
    
    ne = np.matmul(ni[:, 1:], np.arange(1, levels))
    
    # R_tot = np.zeros((M, ))
    # w1 = np.zeros((M, levels - 1))
    # for i in range(levels - 1):
    #     N_recomb[:, i] = N_recomb_tot*R[i](T)
    #     w1[:, i] = ne*ni[:, i + 1]*R[i](T)*dt
    #     R_tot += R[i](T)
    
    # for i in range(levels - 1):
    #     N_recomb[:, i] = np.ceil(N_recomb[:, i]/R_tot)
    #     for j in range(M):
    #         for k in range(int(N_recomb[j, i])):
    #             mu = np.sqrt(rng.random())
    #             start_time = dt*rng.random()
    #             xi = rng.random()
            
    #             nu = Eth[i]/h
    #             photon = Particle(edges[j], mu, nu, w1[j, i], j, (dt - start_time)/dt*timestep_dist)
    #             census.append(photon)
    
    index = 0
    flux = np.zeros((M, ))
    energy_density = np.zeros((M, ))
    dni = np.zeros((M, levels))
    dT = np.zeros((M, ))
    
    while index < len(census):
        p = census[index]
        # For every particle in the system
        while p.timestep_dist > 0 and p.cell < M and p.cell >= 0:
            # Move the particle until it leaves the domain or the timestep ends
            current_cell = p.cell
            
            Gamma = np.zeros((levels - 1, ))
            Gamma_tot = 0
            for i in range(levels - 1):
                Gamma[i] = Gamma_nu[i](h*p.nu)*(h*p.nu >= Eth[i])*ni[current_cell, i]
                Gamma_tot += Gamma[i]
            
            s = p.move(edges[current_cell:(current_cell + 2)], Gamma_tot)
            
            dx = edges[current_cell + 1] - edges[current_cell]
            
            if Gamma_tot < 1e-8:
                dflux = s*p.w/(dx*dt)
            else:
                dflux = 1/(Gamma_tot*dx*dt)*p.w*(1 - np.exp(-Gamma_tot*s))
                
            flux[current_cell] += dflux
            energy_density[current_cell] += dflux*h*p.nu/c
            
            for i in range(levels - 1):
                photoi = min(ni_old[current_cell, i] + dni[current_cell, i], Gamma[i]*dflux*dt)
                
                dni[current_cell, i] -= photoi
                dni[current_cell, i + 1] += photoi
                dT[current_cell] += photoi*(h*p.nu - Eth[i])/(xsf.cv(T, n + ne[current_cell]))
            
            p.reduce_weight(s, Gamma_tot)
        
        # Reset the distance each particle has traveled 
        p.timestep_dist = timestep_dist
            
        if p.w < cutoff or p.cell >= M or p.cell < 0:
            # If the particle has a weight below the cutoff or leaves the domain remove it
            census.pop(index)
        else:
            # Move on to the next particle
            index += 1
    
    for i in range(levels - 1):
        recomb = np.min(np.array([ni_old[:, i + 1] + dni[:, i + 1], ne*ni[:, i + 1]*R[i](T)*dt]), axis=0)
        
        dni[:, i] += recomb
        dni[:, i + 1] -= recomb
        dT -= recomb*(Eth[i] + 1.5*T)/(xsf.cv(T, n + ne))
        
    dne = np.matmul(dni[:, 1:], np.arange(1, levels))
    dT -= 1.5*T*dne/(xsf.cv(T, n + ne))

    return census, flux, dT, dni, energy_density

def iterative_solver(census, ni, T, tol, max_it, emission_rate, seed):
    # Define error metrics and max iterations
    it = 0
    err = np.ones((3, ))*n
    
    # Copy the initial population which can be reused in each iteration
    population = copy.deepcopy(census)
    T_old = T.copy()
    T_previt = T.copy()
    flux = np.zeros((M, ))
    flux_old = flux.copy()
    ni_old = ni.copy()
    ni_previt = ni.copy()
    while (max(abs(err)) > tol and it < max_it):
        census = copy.deepcopy(population)
        census, flux, dT, dni, energy_density = advance_particles(census, ni, T, emission_rate, seed, ni_old)
        
        # Calculate the update to ionization level and associated error
        #dnb = sigma_a*(n - nb)*flux*dt
        ni = ni_old + dni
        
        err[0] = np.linalg.norm(ni - ni_previt)
        ni_previt = ni.copy()
        
        # Calculate the error from flux
        err[1] = np.linalg.norm(flux - flux_old)
        flux_old[:] = flux[:]
        
        # Calculate error from material temperature and update
        T = T_old + dT
        err[2] = np.linalg.norm(T - T_previt)
        T_previt[:] = T[:]
        
        it += 1
        
    return census, ni, flux, T, energy_density

def plot_ionization_level(edges, ni, n, t):
    z_loc = edges[:-1] + np.diff(edges)
    plt.plot(z_loc, ni/n)
    plt.xlabel('z-location [cm]')
    plt.ylabel('Ion fraction')
    plt.legend(['n'+ str(i) for i in range(levels)])
    plt.title('Ion fraction at t='+ str(np.round(t, 2)))
    plt.show()
    
def temperature_plots(edges, energy_denisty, T, t):
    plt.subplot(2, 1, 2)
    z_loc = edges[:-1] + np.diff(edges)
    plt.plot(z_loc, T, label='t='+str(np.round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel('Material temperature [keV]')
    plt.subplot(2, 1, 1)
    plt.plot(z_loc, energy_density, t)
    plt.ylabel('Energy Density [keV/$cm^3$]')
    
def two_state_plots(edges, ni, n, energy_density, t):
    plt.subplot(2, 1, 2)
    z_loc = edges[:-1] + np.diff(edges)
    plt.plot(z_loc, ni[:, 1]/n, label='t='+str(np.round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel('Ion fraction')
    plt.subplot(2, 1, 1)
    plt.plot(z_loc, energy_density, t)
    plt.ylabel('Energy Density [keV/$cm^3$]')
    

census=[]
ni = np.zeros((M, levels))
ni[:, 0] = np.ones((M, ))*n
for t in range(int(t_max/dt) + 1):
    #print("Time step: ", t)
    census, ni, flux, T, energy_density = iterative_solver(census, ni, T, 1e-8*n, 100, emission_rate, t)
    
    if t == 200:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        # ADD CONSISTANT FIGURE/AXIS NUMBER FOR PLOTS
        temperature_plots(edges, energy_density, T, t*dt)
        print(t*dt)
    elif t == 900:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        temperature_plots(edges, energy_density, T, t*dt)
        print(t*dt)
    elif t == 1500:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        temperature_plots(edges, energy_density, T, t*dt)
        print(t*dt)
    elif t == 2000:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        temperature_plots(edges, energy_density, T, t*dt)
        print(t*dt)
    elif t == 4000:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        temperature_plots(edges, energy_density, T, t*dt)
        print(t*dt)

# for t in range(int(t_max/dt) + 1):
#     census, flux = advance_particles(census, Sigma, flux*0, emission_rate, t)

# two_state_plots(edges, ni, n, energy_density, t_max*dt)
temperature_plots(edges, energy_density, T, t_max)
plt.legend()
plt.show()

plot_ionization_level(edges, ni, n, t_max)
plt.show()