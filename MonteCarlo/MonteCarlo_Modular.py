# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:07:35 2025

This script uses a Monte Carlo method to solve a PI Front problem in planar
geometry

@author: jzola2
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append("..")
import CrossSectionFunctions as xsf
from numba.experimental import jitclass
from numba.typed import List
from numba import int64, float64, typeof, jit

from Particle import Particle
from Mesh import Mesh
import Constants

# Problem setup
N_max = 10000                      # Maximum number of particles for Russian Roulette
N_source = 1000                    # Number of particles sourced per iteration
N_recomb_tot = 10                  # Number of particles from recombination per cell
M = 100                             # Number of cells
edges = np.linspace(0, 0.7, M + 1)  # Cell edge location [cm]
flux = np.zeros((M, ))              # Scalar flux in each cell
t_max = 1                           # Maximum time
dt = 0.001                         # Time step [ns]
Ts = 0.1                            # Source temperature [keV]
n = 4.3e20                            # Number density of particles in medium [particles/cm^3]
use_num_time_steps = False               # Use a number of time steps instead of a max time
plot_time = 0.2
plot_interval = 0.2

# Define material 
mat = xsf.Nitrogen()

levels = mat.Z + 1             # Number of ionization levels
# cv = 741/(keV2J*k_B)*(n/(Na)*rho_N)  # Material specific heat [keV/(keV*cm^3)]
# Sigma = np.ones((M, ))*n*sigma_a    # Macroscopic photoionization cross-section

emission_rate = dt*2*np.pi*Constants.sigma_SB*Ts**4/(Constants.keV2GJ)  # Specific intensity of photons emitted [keV/(cm^2)]
tol = 1e-4*np.array([n, emission_rate/(dt*2.71*Ts), Ts])                          # Tolerance for iterative solver

# Timing/profiling variables
moving_particles = [0.0, 0]
calculating_photoi = [0.0, 0]
calculating_recomb = [0.0, 0]
per_iteration = [0.0, 0]
per_time_step = [0.0, 0]
sourcing_particles_boundary = [0.0, 0]
sourcing_particles_recomb = [0.0, 0]
russian_roulette = [0.0, 0]

# Plotting variables
dErr_vec = []
dEpi_vec = []

#@jit
def source_particles(rng, census, mesh, dt, source_loc, w0, N_source, Ts):   
    for i in range(N_source):
        # For each particle to source, parameters
        start_time = dt*rng.random()
        xi = rng.random(5)
        mu = np.sqrt(rng.random())
        nu = xsf.sample_blackbody(Ts, xi)
            
        timestep_dist = Constants.c*dt
        
        photon = Particle(edges[source_loc], mu, nu, w0/(Constants.h*nu), source_loc, (dt - start_time)/dt*timestep_dist)
        census.append(photon)

        mesh.particle_tally[source_loc] += 1
        
    return census, mesh

def advance_particles(census, mesh, *varargs):
    if len(varargs) > 0:
        old_mesh = varargs[0]
    else:
        old_mesh = mesh
    
    mesh.reset_iteration()

    # Move particles through timestep
    time1_move = time.time()
    index = 0
    while index < len(census):
        p = census[index]
        # For every particle in the system
        while p.timestep_dist > 0 and p.cell < M and p.cell >= 0:
            # Use implicit capture to absorb particle and reduce weight, 
            # accounting for photoionization rate
            time1_photoi = time.time()
            mesh.absorb_particle(p, old_mesh, dt)
            time2_photoi = time.time()

            calculating_photoi[0] += time2_photoi - time1_photoi
            calculating_photoi[1] += 1
        
        # Reset the distance each particle has traveled 
        p.timestep_dist = timestep_dist
            
        if p.w < cutoff or p.cell >= M or p.cell < 0:
            # If the particle has a weight below the cutoff or leaves the domain remove it
            census.pop(index)
            if p.w < cutoff and p.cell < M and p.cell >= 0:
                mesh.particle_tally[p.cell] -= 1
        else:
            # Move on to the next particle
            index += 1

    time2_move = time.time()
    moving_particles[0] += time2_move - time1_move
    moving_particles[1] += 1

    # Calculate recombination
    time1_recomb = time.time()
    mesh.calculate_recombination_rate(old_mesh, dt)
    time2_recomb = time.time()
    calculating_recomb[0] += time2_recomb - time1_recomb
    calculating_recomb[1] += 1

    return census, mesh

def iterative_solver(census, mesh, tol, max_it, seed):
    # Define error metrics and max iterations
    it = 0
    err = np.ones((3, ))*n
    err_per_it = np.zeros((3, max_it))
    
    # Copy the initial population which can be reused in each iteration
    #population = List()
    population = []
    for i in range(len(census)):
        population.append(census[i].copy())
    #population = copy.deepcopy(census)

    # Hold data from previous iterations as convergence criteria
    old_mesh = mesh.copy()
    ni_previt = mesh.ni.copy()
    flux_previt = mesh.flux.copy()
    Te_previt = mesh.Te.copy()

    while (np.any(abs(err) > tol) and it < max_it):
        time1_it = time.time()
        #census = List()
        census = []
        for i in range(len(population)):
            census.append(population[i].copy())
        #census = copy.deepcopy(population)

        # Reset particle tally
        mesh.particle_tally[:] = old_mesh.particle_tally[:]

        # Set the random number seed for generating new particles
        rng = np.random.default_rng(seed)

        time1_recomb = time.time()
        # Use previous iteration's temperatures to source particles from recombination
        census = mesh.source_particles_recombination(rng, census, dt)
        time2_recomb = time.time()

        sourcing_particles_recomb[0] += time2_recomb - time1_recomb
        sourcing_particles_recomb[1] += 1

        census, mesh = advance_particles(census, mesh, old_mesh)
        
        # Calculate the update to ionization level and associated error
        mesh.update_state(old_mesh, dt)     # TODO: May be unnecessary. Update dni/dT in place?
        
        err[0] = np.linalg.norm(mesh.ni - ni_previt)
        err_per_it[0, it] = err[0]
        ni_previt = mesh.ni[:]
        
        # Calculate the error from flux
        err[1] = np.linalg.norm(mesh.flux - flux_previt)
        err_per_it[1, it] = err[1]
        flux_previt = mesh.flux[:]
        
        # Calculate error from material temperature and update
        err[2] = np.linalg.norm(mesh.Te - Te_previt)
        err_per_it[2, it] = err[2]
        Te_previt[:] = mesh.Te[:]
        
        it += 1

        time2_it = time.time()
        per_iteration[0] += time2_it - time1_it
        per_iteration[1] += 1

    if it == max_it:
        title = ['Ni err', 'Flux err', 'T err']
        for i in range(3):
            plt.figure(50 + i)
            plt.plot(np.arange(max_it), err_per_it[i, :], label='t=' + str(dt*t))
            plt.title(title[i])
            plt.xlabel("N iterations")
            plt.ylabel("Err")
            

    return census, mesh

# def plot_ionization_level(edges, ni, n, t):
#     z_loc = edges[:-1] + np.diff(edges)
#     plt.plot(z_loc, ni/n)
#     plt.xlabel('z-location [cm]')
#     plt.ylabel('Ion fraction')
#     plt.legend(['n'+ str(i) for i in range(levels)])
#     plt.title('Ion fraction at t='+ str(np.round(t, 2)) + ' ns')
#     plt.show()

# def plot_average_ionization_level(edges, ni, n, t):
#     plt.figure(1)
#     z_loc = edges[:-1] + np.diff(edges)
#     z_bar = np.zeros((len(z_loc), ))
#     for i in range(levels):
#         z_bar += i*ni[:, i]/n
#     plt.plot(z_loc, z_bar, label='t='+str(np.round(t, 2)))
#     plt.xlabel('z-location [cm]')
#     plt.ylabel(r'$ \overline{Z} $')
    
# def temperature_plots(edges, energy_density, T, t):
#     plt.figure(11)
#     plt.subplot(2, 1, 2)
#     z_loc = edges[:-1] + np.diff(edges)
#     plt.plot(z_loc, T, label='t='+str(np.round(t, 2)))
#     plt.xlabel('z-location [cm]')
#     plt.ylabel('Material temperature [keV]')
#     plt.subplot(2, 1, 1)
#     plt.plot(z_loc, (energy_density*Constants.keV2GJ/Constants.a)**0.25, label='t='+str(np.round(t, 2)))
#     plt.ylabel('Radiation temperature [keV]')
    
def two_state_plots(edges, ni, n, energy_density, t):
    plt.subplot(2, 1, 2)
    z_loc = edges[:-1] + np.diff(edges)
    plt.plot(z_loc, ni[:, 1]/n, label='t='+str(np.round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel('Ion fraction')
    plt.subplot(2, 1, 1)
    plt.plot(z_loc, energy_density, t)
    plt.ylabel('Rad. Energy Density [keV/cm$^3$]')

#def alpha_plot(edges, alpha, t):
#    plt.figure(20 + int(t/(1e-3)))
#    z_loc = edges[:-1] + np.diff(edges)
#    plt.plot(z_loc, alpha)
#    plt.title(r'$ \alpha $ at t=' + str(round(t, 2)))
#    plt.xlabel('z-location [cm]')
#    plt.ylabel(r'$ \alpha $ [-]')
#    plt.legend(['n' + str(i) for i in range(levels)])
#
#def write_data(file, t, energy_density, T, ni):
#    f = open(file, 'a')
#    f.write('{:4.2g}'.format(t))
#    f.write('\n\n')
#    form = ''
#    for i in range(M):
#        form += '{:e}, '.format(energy_density[i])
#    f.write(form)
#    f.write('\n\n')
#
#    form = ''
#    for i in range(M):
#        form += '{:e}, '.format(T[i])
#    f.write(form)
#    f.write('\n\n')
#
#    for i in range(M):
#        form = ''
#        for j in range(levels):
#            form += '{:e}, '.format(ni[i, j])
#
#        f.write(form + '\n')
#    
#    f.write('\n')
#    f.close()

# Create particle census
type_photon = Particle(0.1, 0.5, 800.0, 1.0, 0, 0.0)
# census = List.empty_list(typeof(type_photon))
census = []

# Initialize mesh
mesh = Mesh(edges, mat.Z, mat, n, N_recomb_tot)
tot_recomb = np.zeros((M, 100))

# Open file to write to
path = '../Data/'
file_base = path + 'MC_Nitrogen_Reemission_'
index = 0
file = file_base + str(index) + '.txt'
while os.path.exists(file):
    print('File \'' + file + '\' already exists, creating ', end='')
    index += 1
    file = file_base + str(index) + '.txt'
    print('file \'' + file + '\' instead')
f = open(file, 'w')
f.close()

# Begin time stepping loop
if use_num_time_steps:
    steps = 200
else:
    steps = int(t_max/dt) + 1

for t in range(steps):
    # Seed random number generator
    rng = np.random.default_rng(t)
    
    # Boundary source information
    source_loc = int(0)
    w0 = emission_rate/N_source
    timestep_dist = Constants.c*dt
    cutoff = 1e-10*w0

    # Source particles from boundary condition for each time step
    time1 = time.time()
    census, mesh = source_particles(rng, census, mesh, dt, source_loc, w0, N_source, Ts)
    time2 = time.time()
    sourcing_particles_boundary[0] += time2 - time1
    sourcing_particles_boundary[1] += 1

    # Invoke iterative solver to converge ni, T
    time1 = time.time()
    census, mesh = iterative_solver(census, mesh, tol, 100, t)
    time2 = time.time()
    per_time_step[0] += time2 - time1
    per_time_step[1] += 1

    dErr_vec.append(mesh.dErr[0])
    dEpi_vec.append(mesh.dEpi[0])

    #tot_recomb[:, t] = np.sum(mesh.recomb_rate, axis=1)

    time1 = time.time()
    # Roulette particles to reduce particle counts
    rng = np.random.default_rng(100000 + t)

    roulette_pool = np.zeros((M, ))

    # Roulette particles
    kill_prob = N_max/len(census)

    if kill_prob < 1:
        index = 0
        while index < len(census):
            if rng.random() > kill_prob and mesh.particle_tally[census[index].cell] > 1:
                dead_particle = census.pop(index)
                roulette_pool[dead_particle.cell] += Constants.h*dead_particle.nu*dead_particle.w
                mesh.particle_tally[dead_particle.cell] -= 1
            else:
                index += 1

        for particle in census:
            particle.roulette(roulette_pool, mesh.particle_tally)

    roulette_pool = np.zeros((M, ))
    time2 = time.time()
    russian_roulette[0] += time2 - time1
    russian_roulette[1] += 1

    if t*dt > plot_time:
        mesh.plot_temperatures(t*dt)
        mesh.plot_average_ionization_level(t*dt)
        mesh.plot_alpha(t*dt)
        mesh.write_data(file, t*dt)
        print(t*dt)
        plot_time += plot_interval

#     if t == 100:
#         # two_state_plots(edges, ni, n, energy_density, t*dt)
#         # ADD CONSISTANT FIGURE/AXIS NUMBER FOR PLOTS
#         temperature_plots(edges, mesh.energy_density, mesh.Te, t*dt)
#         plot_average_ionization_level(edges, mesh.ni, n, t*dt)
#         alpha_plot(edges, mesh.alpha(), t*dt)
#         write_data(file, np.round(t*dt, 2), mesh.energy_density, mesh.Te, mesh.ni)
#         print(t*dt)
#     elif t == 450:
#         # two_state_plots(edges, ni, n, energy_density, t*dt)
#         temperature_plots(edges, mesh.energy_density, mesh.Te, t*dt)
#         plot_average_ionization_level(edges, mesh.ni, n, t)
#         alpha_plot(edges, mesh.alpha(), t*dt)
#         write_data(file, np.round(t*dt, 2), mesh.energy_density, mesh.Te, mesh.ni)
#         print(t*dt)
#     elif t == 750:
#         # two_state_plots(edges, ni, n, energy_density, t*dt)
#         temperature_plots(edges, mesh.energy_density, mesh.Te, t*dt)
#         plot_average_ionization_level(edges, mesh.ni, n, t)
#         alpha_plot(edges, mesh.alpha(), t*dt)
#         write_data(file, np.round(t*dt, 2), mesh.energy_density, mesh.Te, mesh.ni)
#         print(t*dt)
#     elif t == 1000:
#         # two_state_plots(edges, ni, n, energy_density, t*dt)
#         temperature_plots(edges, mesh.energy_density, mesh.Te, t*dt)
#         plot_average_ionization_level(edges, mesh.ni, n, t)
#         alpha_plot(edges, mesh.alpha(), t*dt)
#         write_data(file, np.round(t*dt, 2), mesh.energy_density, mesh.Te, mesh.ni)
#         print(t*dt)
#     elif t == 2000:
#         # two_state_plots(edges, ni, n, energy_density, t*dt)
#         temperature_plots(edges, mesh.energy_density, mesh.Te, t*dt)
#         plot_average_ionization_level(edges, mesh.ni, n, t)
#         write_data(file, np.round(t*dt, 2), mesh.energy_density, mesh.Te, mesh.ni)
#         print(t*dt)

# two_state_plots(edges, ni, n, energy_density, t_max*dt)
plt.figure(11)
mesh.plot_temperatures(t_max)
plt.legend()
plt.show()

mesh.plot_average_ionization_level(t_max)
plt.legend()
plt.show()

mesh.plot_ionization_level(t_max)
plt.show()

mesh.write_data(file, np.round(t*dt, 2))

print("Time spent:")
print("Moving particles: ", end='')
print(moving_particles[0]/moving_particles[1])
print("Computing Photoionization: ", end='')
print(calculating_photoi[0]/calculating_photoi[1])
print("Computing Recombination: ", end='')
print(calculating_recomb[0]/calculating_recomb[1])
print("On each iteration: ", end='')
print(per_iteration[0]/per_iteration[1])
print("Per time step: ", end='')
print(per_time_step[0]/per_time_step[1])
print("Sourcing particles on boundary: ", end='')
print(sourcing_particles_boundary[0]/sourcing_particles_boundary[1])
print("Sourcing particles from recombination: ", end='')
print(sourcing_particles_recomb[0]/sourcing_particles_recomb[1])
print("Russian rouletting particles: ", end='')
print(russian_roulette[0]/russian_roulette[1])

time = np.linspace(0, t_max, steps)
plt.plot(time, dEpi_vec, label='$dE_{pi}$')
plt.plot(time, dErr_vec, label='$dE_{rr}$')
plt.legend()
plt.show()

# f = open(file, 'a')
# f.write("Recombination in first 100 time steps: \n")
# for i in range(100):
#     for j in range(M):
#         f.write('{:e}, '.format(tot_recomb[j, i]))
#     f.write('\n')