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
from numba import float64, typeof, jit

from Particle import Particle
from Mesh import Mesh
import PlottingFunctions as plot
import Constants
import json

# Read input file
file = open('Inputs/Gray.json', 'r')
inputs = json.load(file)

# Construct file to write out to
path = '../Data/'
file_base = path + 'MC_'
file_base += inputs["Name"] + "_"

# Assign simulation parameters from data file
N_max = inputs["Max Particles"]     # Maximum number of particles for Russian Roulette
file_base += str(N_max) + "MaxPart_"

# Initialize mesh
mesh_params = inputs["Mesh"]

if mesh_params["Material"].lower() == "nitrogen":
    # Define material 
    mat = xsf.Nitrogen()
    file_base += mesh_params["Material"] + "_"
else:
    print("Materials other than Nitrogen not currently supported! Aborting...")
    assert 0

# TODO: Change mesh class to take array input for multiple dimensions
M = mesh_params["Number of Cells"][0]      # Number of cells
n = mesh_params["Density"][0]              # Number density of particles in medium [particles/cm^3]
bounds = mesh_params["Domain Bounds"][0]
edges = np.linspace(bounds[0], bounds[1], M + 1)

# Read number of particles sourced from atomic physics in cells
if inputs["Physics"]["Radiative Recombination"] and inputs["Physics"]["Bremsstrahlung"]:
    N_recomb_tot = int(mesh_params["Particles Sourced Per Cell"]/2)
    N_bremsstrahlung = int(mesh_params["Particles Sourced Per Cell"]/2)
    file_base += "Reemission_B_"
elif inputs["Physics"]["Radiative Recombination"]:
    N_recomb_tot = mesh_params["Particles Sourced Per Cell"]
    N_bremsstrahlung = 0
    file_base += "Reemission_"
elif inputs["Physics"]["Bremsstrahlung"]:
    N_recomb_tot = 0
    N_bremsstrahlung = mesh_params["Particles Sourced Per Cell"]
    file_base += "B_"
else:
    N_recomb_tot = 0
    N_bremsstrahlung = 0

if inputs["Physics"]["Electron Impact Ionization"]:
    file_base += "EII_"

mesh = Mesh(edges, mat.Z, mat, n, N_recomb_tot, N_bremsstrahlung)

# Source temperature function [keV]
@jit
def Ts_Const(t):
    return Ts

@jit
def Ts_Ramp(t):
    t_peak = 1
    
    if t < t_peak:
        return (t/t_peak)*Ts
    else:
        return Ts

# TODO: Make a source object which incorporates the source particles function and make an array of source 
# to loop through in main time stepping portion of code
for source in inputs["Sources"]:
    N_source = source["Particles Sourced Per Time Step"]   # Number of particles sourced per iteration
    Ts = source["Temperature"]                             # Peak Source temperature [keV]
    if source["Function"].lower() == "ramp":
        Ts_Function = Ts_Ramp
    elif source["Function"].lower() == "const":
        Ts_Function = Ts_Const

# Import information about time stepping
use_num_time_steps = inputs["Time"]["Use Num Time Steps"]
dt = inputs["Time"]["Time Step"]             # Time step [ns]
file_base += "TimeStep" + str(dt) + "_"
if use_num_time_steps:
    steps = inputs["Time"]["Num Time Steps"]
    t_max = steps*dt
else:
    t_max = inputs["Time"]["Max Time"]           # Maximum time [ns]
    steps = int(t_max/dt) + 1

base_tol = inputs["Time"]["Tolerance"]
#tot_recomb = np.zeros((M, 100))

# Determine Output Information
plot_time = inputs["Outputs"]["Initial Plot Time"]
plot_interval = inputs["Outputs"]["Plot Interval"]

# levels = mat.Z + 1             # Number of ionization levels
# cv = 741/(keV2J*k_B)*(n/(Na)*rho_N)  # Material specific heat [keV/(keV*cm^3)]
# Sigma = np.ones((M, ))*n*sigma_a    # Macroscopic photoionization cross-section
peak_emission_rate = dt*2*np.pi*Constants.sigma_SB*Ts**4/(Constants.keV2GJ)
tol = base_tol*np.array([n, peak_emission_rate/(dt*2.71*Ts), Ts])                          # Tolerance for iterative solver

# Timing/profiling variables
moving_particles = [0.0, 0]
per_iteration = [0.0, 0]
per_time_step = [0.0, 0]
sourcing_particles_boundary = [0.0, 0]
sourcing_particles_recomb = [0.0, 0]
russian_roulette = [0.0, 0]

# Plotting variables
dErr_vec = []
dEpi_vec = []

@jit
def source_particles(rng, census, mesh, dt, source_loc, N_source, Ts, t):

    emission_rate = dt*2*np.pi*Constants.sigma_SB*Ts(t)**4/(Constants.keV2GJ)  # Specific intensity of photons emitted [keV/(cm^2)]
    w0 = emission_rate/N_source
    cutoff = 1e-10*w0
    for i in range(N_source):
        # For each particle to source, parameters
        start_time = dt*rng.random()
        xi = rng.random(5)
        mu = np.sqrt(rng.random())
        nu = xsf.sample_blackbody(Ts(t), xi)
        #nu = Ts/Constants.h
            
        timestep_dist = Constants.c*dt
        
        photon = Particle(mesh.cell_edges[source_loc], mu, nu, w0/(Constants.h*nu), source_loc, (dt - start_time)/dt*timestep_dist)
        census.append(photon)

        mesh.particle_tally[source_loc] += 1
        
    return census, mesh, cutoff

@jit
def advance_particles(census, cutoff, mesh, *varargs):
    if len(varargs) > 0:
        old_mesh = varargs[0]
    else:
        old_mesh = mesh
    
    mesh.reset_iteration()

    # Move particles through timestep
    index = 0
    while index < len(census):
        p = census[index]
        # For every particle in the system
        while p.timestep_dist > 0 and p.cell < M and p.cell >= 0:
            # Use implicit capture to absorb particle and reduce weight, 
            # accounting for photoionization rate
            mesh.absorb_particle(p, old_mesh, dt)
        
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

    # Update the total flux
    # mesh.flux = np.sum(mesh.multigroup_flux, axis=1)

    # Calculate electron impact ionization
    #mesh.calculate_eii(old_mesh, dt) # TODO: Operator split active. Deactivate before fooling around

    # Calculate recombination
    mesh.calculate_rr_rate(old_mesh, dt)
    #mesh.calculate_tbr_rate(old_mesh, dt) # TODO: Operator split active. Deactivate before fooling around

    return census, mesh

def iterative_solver(census, cutoff, mesh, tol, max_it, seed):
    # Define error metrics and max iterations
    it = 0
    err = np.ones((3, ))*n
    err_per_it = np.zeros((3, max_it))
    Z_bar_five_it = np.zeros((mesh.N_cells, 5))
    flux_five_it = np.zeros((mesh.N_cells, 5))
    T_five_it = np.zeros((mesh.N_cells, 5))
    err_per_level = np.zeros((mesh.N_levels, max_it))

    # Copy the initial population which can be reused in each iteration
    population = List()
    #population = []
    for i in range(len(census)):
        population.append(census[i].copy())

    # Hold data from previous iterations as convergence criteria
    old_mesh = mesh.copy()
    mesh_previt = mesh.copy()
    mesh_previt.reset_iteration()

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

        census = mesh.source_particles_bremsstrahlung(rng, census, dt)

        sourcing_particles_recomb[0] += time2_recomb - time1_recomb
        sourcing_particles_recomb[1] += 1

        census, mesh = advance_particles(census, cutoff, mesh, old_mesh)

        # Calculate the update to ionization level and associated error
        mesh.update_state(old_mesh, mesh_previt, dt, (1.0 + 0.0*(it == 0)), False) ##########
        
        err[0] = np.linalg.norm(mesh.ni - mesh_previt.ni)
        err_per_it[0, it] = err[0]
        for i in range(mesh.N_levels):
            err_per_level[i, it] = np.linalg.norm(mesh.ni[i, :] - mesh_previt.ni[i, :])
        
        # Calculate the error from flux
        err[1] = np.linalg.norm(mesh.flux - mesh_previt.flux)
        err_per_it[1, it] = err[1]
        
        # Calculate error from material temperature and update
        err[2] = np.linalg.norm(mesh.Te - mesh_previt.Te)
        err_per_it[2, it] = err[2]

        mesh_previt = mesh.copy()

        if it < 5:
            Z_bar_five_it[:, it] = np.dot(mesh.ni, np.arange(mesh.N_levels + 1))/mesh.atom_density
            flux_five_it[:, it] = mesh.flux.copy()
            T_five_it[:, it] = mesh.Te.copy()
        
        it += 1

        time2_it = time.time()
        per_iteration[0] += time2_it - time1_it
        per_iteration[1] += 1

    if it == max_it:
        print("Max iterations reached at t={0}".format(t))
        print("Ni rel err: {:g}".format(err[0]/tol[0]))
        print("Flux rel err: {:g}".format(err[1]/tol[1]))
        print("T rel err {:g} \n".format(err[2]/tol[2]))

    #    plt.figure(55)
    #    plt.plot(np.arange(max_it), err_per_level.T)
    #    plt.xlabel("N iterations")
    #    plt.ylabel("Err")
    #    plt.legend([str(level) for level in range(mesh.N_levels + 1)])
    #    plt.title("Error per level")
#
    #    for i in range(3):
    #        plt.figure(50 + i)
    #        plt.subplot(2, 1, 2)
    #        plt.semilogy(np.arange(max_it), err_per_it[i, :], label='t=' + str(dt*t))
    #        plt.xlabel("N iterations")
    #        plt.ylabel("Err")
#
    #    symbols = ['o', 'v', '^', '<', '>']
#
    #    for i in range(5):
    #        plt.figure(51)
    #        plt.subplot(2, 1, 1)
    #        plt.plot(mesh.cell_centers, flux_five_it[:, i], label=str(i))
    #        plt.title('Flux')
    #        plt.figure(52)
    #        plt.subplot(2, 1, 1)
    #        plt.plot(mesh.cell_centers, T_five_it[:, i], label=str(i))
    #        plt.title('Temperature')
    #        plt.figure(50)
    #        plt.subplot(2, 1, 1)
    #        plt.plot(mesh.cell_centers, Z_bar_five_it[:, i], symbols[i], label=str(i))
    #        plt.title(r'$\overline{Z}$')
    #    
    #    plt.figure(50)
    #    plt.legend()
    #    plt.figure(51)
    #    plt.legend()
    #    plt.figure(52)
    #    plt.legend()
    #    plt.show()

    # TODO: Operator split - perform eii and tbr calculations after converging on photoionization
    mesh.dni[:] = np.zeros((mesh.N_cells, mesh.N_levels + 1))
    mesh.calculate_eii(mesh, dt)
    mesh.calculate_tbr_rate(mesh, dt)
    mesh.ni = np.maximum(0.0, mesh.ni + mesh.dni)
    mesh.ne = np.dot(mesh.ni, np.arange(mesh.N_levels + 1))
    if np.any(mesh.ni < 0) or np.any(mesh.ni != mesh.ni):
        print("Negative ni:")
        print("Time step: ", end="")
        print(t)
        print("Ne")
        print(mesh.ne)
        print("ni")
        print(mesh.ni)
        print("dni")
        print(mesh.dni)
        
        assert 0
    mesh.dni[:] = np.zeros((mesh.N_cells, mesh.N_levels + 1)) # TODO: Part of operator split

    return census, mesh
    
def two_state_plots(edges, ni, n, energy_density, t):
    plt.subplot(2, 1, 2)
    z_loc = edges[:-1] + np.diff(edges)
    plt.plot(z_loc, ni[:, 1]/n, label='t='+str(np.round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel('Ion fraction')
    plt.subplot(2, 1, 1)
    plt.plot(z_loc, energy_density, t)
    plt.ylabel('Rad. Energy Density [keV/cm$^3$]')

# Create particle census
type_photon = Particle(0.1, 0.5, 800.0, 1.0, 0, 0.0)
census = List.empty_list(typeof(type_photon))
#census = []

# Open file to write to
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
for t in range(steps):
    # Seed random number generator
    rng = np.random.default_rng(t)
    
    # Boundary source information
    source_loc = int(0)
    timestep_dist = Constants.c*dt

    # Source particles from boundary condition for each time step
    time1 = time.time()
    census, mesh, cutoff = source_particles(rng, census, mesh, dt, source_loc, N_source, Ts_Function, (t + 1)*dt)
    time2 = time.time()
    sourcing_particles_boundary[0] += time2 - time1
    sourcing_particles_boundary[1] += 1

    if t == 0:
        plot.plot_radiation_spectrum(mesh, census, 0.0035, 0, 0)
        plt.show()

    # Invoke iterative solver to converge ni, T
    time1 = time.time()
    census, mesh = iterative_solver(census, cutoff, mesh, tol, 100, t)
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

        rad_energy_per_cell_bfr = np.zeros((M, ))
        rad_energy_per_cell_aft = np.zeros((M, ))
        for particle in census:
            rad_energy_per_cell_bfr[particle.cell] += particle.w*Constants.h*particle.nu

        while index < len(census):
            if rng.random() > kill_prob and mesh.particle_tally[census[index].cell] > 1:
                dead_particle = census.pop(index)
                roulette_pool[dead_particle.cell] += Constants.h*dead_particle.nu*dead_particle.w
                mesh.particle_tally[dead_particle.cell] -= 1
            else:
                index += 1

        for particle in census:
            particle.roulette(roulette_pool, mesh.particle_tally)
            rad_energy_per_cell_aft[particle.cell] += particle.w*Constants.h*particle.nu

    roulette_pool = np.zeros((M, ))
    time2 = time.time()
    russian_roulette[0] += time2 - time1
    russian_roulette[1] += 1

    if t*dt >= plot_time:
        plot.plot_temperatures(mesh, t*dt)
        plot.plot_average_ionization_level(mesh, t*dt)
        plot.plot_radiation_spectrum(mesh, census, 0.0035, t*dt, t*dt > t_max - plot_interval)
        plot.plot_radiation_spectrum(mesh, census, 0.2, t*dt, t*dt > t_max - plot_interval)
        plot.plot_radiation_spectrum(mesh, census, 0.35, t*dt, t*dt > t_max - plot_interval)
        plot.plot_alpha(mesh, t*dt)
        plot.write_data(mesh, file, t*dt)
        print(t*dt)
        plot_time += plot_interval

# two_state_plots(edges, ni, n, energy_density, t_max*dt)
plt.figure(11)
#plot.plot_temperatures(mesh, t_max)
plt.legend()
plt.figure(22)
plt.legend()
plt.show()

#plt.figure(22)
#plot.plot_average_ionization_level(mesh, t_max)
#plt.legend()
#plt.show()

#plot.plot_ionization_level(mesh, t_max)
#plt.show()

plot.write_data(mesh, file, np.round(t*dt, 2))

print("Time spent:")
#print("Moving particles: ", end='')
#print(moving_particles[0]/moving_particles[1])
#print("Computing Photoionization: ", end='')
#print(calculating_photoi[0]/calculating_photoi[1])
#print("Computing Recombination: ", end='')
#print(calculating_recomb[0]/calculating_recomb[1])
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