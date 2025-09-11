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
from numba.experimental import jitclass
from numba.typed import List
from numba import int64, float64, typeof, jit

from Particle import Particle
from Mesh import Mesh
import Constants

# Physical constants
a = 0.01372                         # Source spectrum [GJ/(cm^3*keV^4)]
c = 30                              # Speed of light [cm/ns]
sigma_SB = a*c/(4*np.pi)            # Stefan-Boltzmann constant [GJ/(cm^2*ns*ster*keV^4)]
keV2GJ = 1.602e-25                  # Conversion factor from keV to GJ
h = 4.415e-9                        # Planck's constant [keV-ns]
keV2J = 1.602e-16                   # Conversion factor from keV to J
k_B = 8.617e-8                      # Boltzmann constant [keV/K]
Na = 6.6022e23                      # Avogadro's number [#/mol]
rho_N = 0.02802                     # Molar mass of Nitrogen [kg/mol]

# Problem setup
N_max = 10000                      # Maximum number of particles for Russian Roulette
N_source = 1000                    # Number of particles sourced per iteration
N_recomb_tot = 10                  # Number of particles from recombination per cell
M = 100                             # Number of cells
edges = np.linspace(0, 0.7, M + 1)  # Cell edge location [cm]
flux = np.zeros((M, ))              # Scalar flux in each cell
T = np.ones((M, ))*1e-5             # Material temperature in each cell [keV]
t_max = 1                           # Maximum time
dt = 0.001                         # Time step [ns]
Ts = 0.1                            # Source temperature [keV]
n = 4.3e20                            # Number density of particles in medium [particles/cm^3]
tol = 1e-4                          # Tolerance for iterative solver
particle_tally = np.zeros((M, ))    # A global container which counts the number of particles in each cell
particle_tally_backup = np.zeros((M, )) # A global backup for the particle tally for use with the iterative solver

# Define material 
mat = xsf.Nitrogen()

Gamma_nu = [mat.pi_n1, mat.pi_n2, mat.pi_n3, mat.pi_n4, mat.pi_n5, mat.pi_n6, mat.pi_n7]  # Photoionization microscopic cross-section [cm^2]
R = [mat.rr_n1, mat.rr_n2, mat.rr_n3, mat.rr_n4, mat.rr_n5, mat.rr_n6, mat.rr_n7]      # Radiative recombination rate constant [cm^2/s]
levels = len(Gamma_nu) + 1             # Number of ionization levels
Eth = mat.Eth   # Threshhold energy [keV]
N_recomb = np.zeros((M, levels - 1)) # The number of particles to source from each ionization level
# cv = 741/(keV2J*k_B)*(n/(Na)*rho_N)  # Material specific heat [keV/(keV*cm^3)]
#Sigma = np.ones((M, ))*n*sigma_a    # Macroscopic photoionization cross-section

emission_rate = dt*sigma_SB*Ts**4/(keV2GJ)  # Specific intensity of photons emitted [keV/(cm^2)]

@jit
def source_particles(census, particle_tally, rng, dt, source_loc, w0, number, **kwargs):
    flag = 0
    for key, value in kwargs.items():
        # Determine how to source particles: via blackbody radiation or radiative recombination
        if key == 'blackbody':
            # The keyword argument is a blackbody temperature
            Ts = value
            flag = 1
        elif key == 'recombination':
            # The keyword argument is a threshhold energy
            Eth = value
            flag = 2
        elif key == 'temperature':
            # The keyword argument is an electron temperature
            Te = value
    
    if flag == 0:
        # Default: blackbody radiation with 0.1 keV source temperature
        flag = 1
        Ts = 0.1
    
    for i in range(number):
        # For each particle to source, parameters
        start_time = dt*rng.random()
        xi = rng.random(5)
        
        if flag == 1:
            mu = np.sqrt(rng.random())
            nu = xsf.sample_blackbody(Ts, xi)
        elif flag == 2:
            mu = 2*rng.random() - 1
            nu = (Eth + xsf.sample_maxwellian(T, xi[0]))/h
            
        timestep_dist = c*dt
        
        photon = Particle(edges[source_loc], mu, nu, w0/(h*nu), source_loc, (dt - start_time)/dt*timestep_dist)
        census.append(photon)

        particle_tally[source_loc] += 1
        
    return census

def advance_particles(census, particle_tally, ni, T, emission_rate, seed, *varargs):
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
        xi = rng.random(5)
        
        nu = xsf.sample_blackbody(Ts, xi)
        photon = Particle(edges[source_loc], mu, nu, w0/(h*nu), source_loc, (dt - start_time)/dt*timestep_dist)
        census.append(photon)
        particle_tally[source_loc] += 1
    
    ne = np.matmul(ni[:, 1:], np.arange(1, levels))
    
    # Source particles from recombination
    R_tot = np.zeros((M, ))
    w1 = np.zeros((M, levels - 1))
    for i in range(levels - 1):
        N_recomb[:, i] = N_recomb_tot*R[i](T)*(T > 1e-5)
        w1[:, i] = ne*ni[:, i + 1]*R[i](T)*dt
        R_tot += R[i](T)
    
    for i in range(levels - 1):
        N_recomb[:, i] = np.ceil(N_recomb[:, i]/R_tot)
        for j in range(M):
            for k in range(int(N_recomb[j, i])):
                mu = np.sqrt(rng.random())
                start_time = dt*rng.random()
                xi = rng.random()
            
                nu = (Eth[i] + xsf.sample_maxwellian(T[j], xi))/h
                photon = Particle(edges[j], mu, nu, w1[j, i]/int(N_recomb[j, i]), j, (dt - start_time)/dt*timestep_dist)
                census.append(photon)

                particle_tally[j] += 1
    
    index = 0
    flux = np.zeros((M, ))
    energy_density = np.zeros((M, ))
    dni = np.zeros((M, levels))
    alpha = np.zeros((M, levels - 1))

    # Define energy/temperatures
    dEpi = np.zeros((M, ))
    dErr = np.zeros((M, ))
    dEsp = np.zeros((M, ))
    dT = np.zeros((M, ))
    tot_photoi = np.zeros((M, levels - 1))

    # Move particles through timestep
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
            
            s, particle_tally = p.move(edges[current_cell:(current_cell + 2)], particle_tally)
            
            dx = edges[current_cell + 1] - edges[current_cell]
            
            if Gamma_tot < 1e-8:
                dflux = s*p.w/(dx*dt)
            else:
                dflux = 1/(Gamma_tot*dx*dt)*p.w*(1 - np.exp(-Gamma_tot*s))
                
            flux[current_cell] += dflux
            energy_density[current_cell] += dflux*h*p.nu/c
            
            for i in range(levels - 1):
                # Calculate the number of photoionized particles for this ionization state
                # If this would photoionize more than remaining particles, reduce to zero instead
                photoi = min(ni_old[current_cell, i] + dni[current_cell, i], Gamma[i]*dflux*dt)
                
                dni[current_cell, i] -= photoi
                dni[current_cell, i + 1] += photoi
                tot_photoi[current_cell, i] += photoi
                dEpi[current_cell] += photoi*(h*p.nu - Eth[i])
            
            p.reduce_weight(s, Gamma_tot)
        
        # Reset the distance each particle has traveled 
        p.timestep_dist = timestep_dist
            
        if p.w < cutoff or p.cell >= M or p.cell < 0:
            # If the particle has a weight below the cutoff or leaves the domain remove it
            census.pop(index)
            if p.w < cutoff and p.cell < M and p.cell >= 0:
                particle_tally[p.cell] -= 1
        else:
            # Move on to the next particle
            index += 1
    
    # Calculate recombination
    for i in range(levels - 1):
        # Calculate the number of ions which recombine at this temperature
        recomb = np.min(np.array([ni_old[:, i + 1] + dni[:, i + 1], ne*ni[:, i + 1]*R[i](T)*dt]), axis=0)
        
        dni[:, i] += recomb
        dni[:, i + 1] -= recomb
        dErr += recomb*(Eth[i] + 1.5*T)

        for j in range(M):
            alpha[j, i] = 0 if tot_photoi[j, i] == 0 else recomb[j]/tot_photoi[j, i]

    # Update ionization state and temperature    
    dne = np.matmul(dni[:, 1:], np.arange(1, levels))
    for i in range(M):
        dEsp[i] = mat.E_spectral(T[i], ni[i, :])*dt

    dT += (dEpi - dErr - dEsp - 1.5*T*dne)/(xsf.cv(T, n + ne))

    return census, particle_tally, flux, dT, dni, energy_density, alpha

def iterative_solver(census, particle_tally, ni, T, tol, max_it, emission_rate, seed):
    # Define error metrics and max iterations
    it = 0
    err = np.ones((3, ))*n
    
    # Copy the initial population which can be reused in each iteration
    #population = List()
    population = []
    for i in range(len(census)):
        population.append(census[i].copy())
    #population = copy.deepcopy(census)
    T_old = T.copy()
    T_previt = T.copy()
    flux = np.zeros((M, ))
    flux_old = flux.copy()
    ni_old = ni.copy()
    ni_previt = ni.copy()
    particle_tally_backup = particle_tally.copy()
    while (max(abs(err)) > tol and it < max_it):
        #census = List()
        census = []
        for i in range(len(population)):
            census.append(population[i].copy())
        #census = copy.deepcopy(population)
        particle_tally = particle_tally_backup.copy()
        census, particle_tally, flux, dT, dni, energy_density, alpha = advance_particles(census, particle_tally, ni, T, emission_rate, seed, ni_old)
        
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
        
    return census, particle_tally, ni, flux, T, energy_density, alpha

def plot_ionization_level(edges, ni, n, t):
    z_loc = edges[:-1] + np.diff(edges)
    plt.plot(z_loc, ni/n)
    plt.xlabel('z-location [cm]')
    plt.ylabel('Ion fraction')
    plt.legend(['n'+ str(i) for i in range(levels)])
    plt.title('Ion fraction at t='+ str(np.round(t, 2)) + ' ns')
    plt.show()

def plot_average_ionization_level(edges, ni, n, t):
    plt.figure(1)
    z_loc = edges[:-1] + np.diff(edges)
    z_bar = np.zeros((len(z_loc), ))
    for i in range(levels):
        z_bar += i*ni[:, i]/n
    plt.plot(z_loc, z_bar, label='t='+str(np.round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel(r'$ \overline{Z} $')
    
def temperature_plots(edges, energy_denisty, T, t):
    plt.figure(11)
    plt.subplot(2, 1, 2)
    z_loc = edges[:-1] + np.diff(edges)
    plt.plot(z_loc, T, label='t='+str(np.round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel('Material temperature [keV]')
    plt.subplot(2, 1, 1)
    plt.plot(z_loc, (energy_density*keV2GJ/a)**0.25, label='t='+str(np.round(t, 2)))
    plt.ylabel('Radiation temperature [keV]')
    
def two_state_plots(edges, ni, n, energy_density, t):
    plt.subplot(2, 1, 2)
    z_loc = edges[:-1] + np.diff(edges)
    plt.plot(z_loc, ni[:, 1]/n, label='t='+str(np.round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel('Ion fraction')
    plt.subplot(2, 1, 1)
    plt.plot(z_loc, energy_density, t)
    plt.ylabel('Rad. Energy Density [keV/cm$^3$]')

def alpha_plot(edges, alpha, t):
    plt.figure(22)
    z_loc = edges[:-1] + np.diff(edges)
    plt.plot(z_loc, alpha)
    plt.title(r'$ \alpha $ at t=' + str(round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel(r'$ \alpha $ [-]')
    plt.legend(['n' + str(i) for i in range(levels)])

def write_data(file, t, energy_density, T, ni):
    f = open(file, 'a')
    f.write('{:4.2g}'.format(t))
    f.write('\n\n')
    form = ''
    for i in range(M):
        form += '{:e}, '.format(energy_density[i])
    f.write(form)
    f.write('\n\n')

    form = ''
    for i in range(M):
        form += '{:e}, '.format(T[i])
    f.write(form)
    f.write('\n\n')

    for i in range(M):
        form = ''
        for j in range(levels):
            form += '{:e}, '.format(ni[i, j])

        f.write(form + '\n')
    
    f.write('\n')
    f.close()

type_photon = Particle(0.1, 0.5, 800.0, 1.0, 0, 0.0)
# census = List.empty_list(typeof(type_photon))
census = []
ni = np.zeros((M, levels))
ni[:, 0] = np.ones((M, ))*n
path = '../Data/'
file = path + 'MC_Nitrogen_Reemission_t=' + str(round(t_max, 2)) + '.txt'
f = open(file, 'w')
f.close()
for t in range(int(t_max/dt) + 1):
    #print("Time step: ", t)
    census, particle_tally, ni, flux, T, energy_density, alpha = iterative_solver(census, particle_tally, ni, T, tol*n, 100, emission_rate, t)
    rng = np.random.default_rng()

    roulette_pool = np.zeros((M, ))

    # Roulette particles
    kill_prob = N_max/len(census)

    if kill_prob < 1:
        index = 0
        while index < len(census):
            if rng.random() > kill_prob and particle_tally[census[index].cell] > 1:
                dead_particle = census.pop(index)
                roulette_pool[dead_particle.cell] += h*dead_particle.nu*dead_particle.w
                particle_tally[dead_particle.cell] -= 1
            else:
                index += 1

        for particle in census:
            particle.roulette(roulette_pool, particle_tally)

    roulette_pool = np.zeros((M, ))
    
    if t == 100:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        # ADD CONSISTANT FIGURE/AXIS NUMBER FOR PLOTS
        temperature_plots(edges, energy_density, T, t*dt)
        plot_average_ionization_level(edges, ni, n, t*dt)
        alpha_plot(edges, alpha, t*dt)
        write_data(file, np.round(t*dt, 2), energy_density, T, ni)
        print(t*dt)
    elif t == 450:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        temperature_plots(edges, energy_density, T, t*dt)
        plot_average_ionization_level(edges, ni, n, t)
        write_data(file, np.round(t*dt, 2), energy_density, T, ni)
        print(t*dt)
    elif t == 750:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        temperature_plots(edges, energy_density, T, t*dt)
        plot_average_ionization_level(edges, ni, n, t)
        write_data(file, np.round(t*dt, 2), energy_density, T, ni)
        print(t*dt)
    elif t == 1000:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        temperature_plots(edges, energy_density, T, t*dt)
        plot_average_ionization_level(edges, ni, n, t)
        write_data(file, np.round(t*dt, 2), energy_density, T, ni)
        print(t*dt)
    elif t == 2000:
        # two_state_plots(edges, ni, n, energy_density, t*dt)
        temperature_plots(edges, energy_density, T, t*dt)
        plot_average_ionization_level(edges, ni, n, t)
        write_data(file, np.round(t*dt, 2), energy_density, T, ni)
        print(t*dt)

# for t in range(int(t_max/dt) + 1):
#     census, flux = advance_particles(census, Sigma, flux*0, emission_rate, t)

# two_state_plots(edges, ni, n, energy_density, t_max*dt)
plt.figure(11)
temperature_plots(edges, energy_density, T, t_max)
plt.legend()
plt.show()

plot_average_ionization_level(edges, ni, n, t)
plt.legend()
plt.show()

plot_ionization_level(edges, ni, n, t_max)
plt.show()

write_data(file, np.round(t*dt, 2), energy_density, T, ni)