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
dim = mesh_params["Dimension"]
M = np.zeros((dim, ), dtype=np.int32)
bounds = np.zeros((dim, 2))
for i in range(dim):
    M[i] = mesh_params["Number of Cells"][i]
    bounds[i] = mesh_params["Domain Bounds"][i]

# Initialize boundary conditions to invalid codes for error detection
bcs = -1*np.ones((dim*2, ), dtype=np.int32)
bc_idx = 0
for pair in mesh_params["Boundary Conditions"]:
    match pair[0].lower():
        case "vacuum":
            bcs[bc_idx] = 0
        case "reflecting":
            bcs[bc_idx] = 1

    match pair[1].lower():
        case "vacuum":
            bcs[bc_idx + 1] = 0
        case "reflecting":
            bcs[bc_idx + 1] = 1

    bc_idx += 2

n = mesh_params["Density"][0]              # Number density of particles in medium [particles/cm^3]
edges = np.zeros((np.sum(M) + dim, ))
start = 0
end = M[0] + 1
for i in range(dim):
    if mesh_params["Cell Spacing"][i].lower() == "linear":
        edges[start:end] = np.linspace(bounds[i, 0], bounds[i, 1], M[i] + 1)
    elif mesh_params["Cell Spacing"][i].lower() == "salzmann":
        edges[start + 1] = 5e-4
        width = 5e-4
        for j in range(M[i] - 1):
            edges[j + start + 2] = edges[j + start + 1] + 1.07*width
            width *= 1.07
    elif mesh_params["Cell Spacing"][i].lower() == "logarithmic":
        # Log space the first 10% of cells
        M1 = int(np.floor(M[i]*0.2))
        M2 = M[i] - M1 + 1
        edges[start] = bounds[start]
        edges[(start + 1):(start + M1 + 1)] = np.logspace(-5, np.log(bounds[1]*0.2)/np.log(10), M1)
        edges[(start + M1):end] = np.linspace(bounds[1]*0.2, bounds[1], M2)

    if i < dim - 1:
        start += end
        end += M[i + 1] + 1

# Read number of particles sourced from atomic physics in cells
use_rr = inputs["Physics"]["Radiative Recombination"]
use_b = inputs["Physics"]["Bremsstrahlung"]
use_eii = inputs["Physics"]["Electron Impact Ionization"]

if use_rr and use_b:
    N_recomb_tot = int(mesh_params["Particles Sourced Per Cell"]/2)
    N_bremsstrahlung = int(mesh_params["Particles Sourced Per Cell"]/2)
    file_base += "Reemission_B_"
elif use_rr:
    N_recomb_tot = mesh_params["Particles Sourced Per Cell"]
    N_bremsstrahlung = 0
    file_base += "Reemission_"
elif use_b:
    N_recomb_tot = 0
    N_bremsstrahlung = mesh_params["Particles Sourced Per Cell"]
    file_base += "B_"
else:
    N_recomb_tot = 0
    N_bremsstrahlung = 0

if use_eii:
    file_base += "EII_"

mesh = Mesh(dim, edges, mat.Z, M, mat, n, N_recomb_tot, N_bremsstrahlung, bcs)

# TODO: Make a source object which incorporates the source particles function and make an array of source 
# to loop through in main time stepping portion of code
for source in inputs["Sources"]:
    N_source = source["Particles Sourced Per Time Step"]   # Number of particles sourced per iteration
    Ts = source["Temperature"]                             # Peak Source temperature [keV]

    # Source temperature function [keV]
    if source["Function"].lower() == "ramp":
        t_peak = source["Peak Time"]
        @jit
        def Ts_Function(t):
            if t < t_peak:
                return (t/t_peak)*Ts
            else:
                return Ts
    elif source["Function"].lower() == "const":
        @jit
        def Ts_Function(t):
            return Ts

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

# Determine Output Information
plot_time = inputs["Outputs"]["Initial Plot Time"]
plot_interval = inputs["Outputs"]["Plot Interval"]
plot_initial_energy_spectrum = inputs["Outputs"]["Plot Initial Spectrum"]
be_verbose = inputs["Outputs"]["Verbose"]

# TODO: Multiply by emission area for source
x_edges = mesh.get_edges_dim(1)
peak_emission_rate = (x_edges[-1] - x_edges[0])*dt*2*np.pi*Constants.sigma_SB*Ts**4/(Constants.keV2GJ)
tol = base_tol*np.array([n, peak_emission_rate/(dt*2.71*Ts), Ts])                          # Tolerance for iterative solver

# Input checking
if be_verbose:
    print("N_source: {0}".format(N_source))
    print("Source Temperature: {:3.1g}".format(Ts))
    print("Peak Time: {:3.1g}".format(t_peak))
    print("Function at t=0: {:3.1g}".format(Ts_Function(0)))
    print("Function at t=0.5*t_peak: {:3.2g}".format(Ts_Function(0.5*t_peak)))
    print("Function at t=t_peak: {:3.1g}".format(Ts_Function(t_peak)))
    print("Function at t=2*t_peak: {:3.1g}".format(Ts_Function(2*t_peak)))
    print("Z Edges: ")
    print(mesh.get_edges_dim(0))
    print("X Edges: ")
    print(mesh.get_edges_dim(1))
    print("Peak Emission Rate: {:g}".format(peak_emission_rate))

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
    # TODO: Add a function like this for each type of source in our source class
    x_edges = mesh.get_edges_dim(1)
    z_edges = mesh.get_edges_dim(0)
    cell_area = (x_edges[-1] - x_edges[0])
    emission_rate = cell_area*dt*2*np.pi*Constants.sigma_SB*Ts(t)**4/(Constants.keV2GJ)  # Specific intensity of photons emitted [keV/(cm^2)]
    w0 = emission_rate/N_source
    cutoff = 1e-10*w0
    for i in range(N_source):
        # For each particle to source, sample parameters
        start_time = dt*rng.random()
        xi = rng.random(5)
        x = rng.random()*(x_edges[-1] - x_edges[0]) + x_edges[0]
        cell_i = np.searchsorted(x_edges, x) - 1
        mu = np.sqrt(rng.random())
        phi = rng.random()*2*np.pi
        nu = xsf.sample_blackbody(Ts(t), xi)
            
        timestep_dist = Constants.c*dt
        
        # Construct the photon and add to census, counting particles in each cell
        photon = Particle(x, z_edges[source_loc], mu, phi, nu, w0/(Constants.h*nu), cell_i, source_loc, (dt - start_time)/dt*timestep_dist)
        census.append(photon)

        source_index = mesh.get_index(cell_i, source_loc)
        mesh.particle_tally[source_index] += 1
        
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
        outside_bounds = False
        while p.timestep_dist > 0 and not outside_bounds:
            # Use implicit capture to absorb particle and reduce weight, 
            # accounting for photoionization rate
            outside_bounds = mesh.absorb_particle(p, old_mesh, use_b, dt)
        
        # Reset the distance each particle has traveled 
        p.timestep_dist = timestep_dist
            
        # TODO: Update to evaluate boundary conditions
        if p.w < cutoff or outside_bounds:
            # If the particle has a weight below the cutoff or leaves the domain remove it
            census.pop(index)
            if p.w < cutoff and not outside_bounds:
                mesh.particle_tally[mesh.get_index(p.cell_i, p.cell_k)] -= 1
        else:
            # Move on to the next particle
            index += 1

    # Update the total energy density
    # mesh.energy_density = np.sum(mesh.multigroup_flux, axis=1)

    # Calculate electron impact ionization
    #mesh.calculate_eii(old_mesh, dt) # TODO: Operator split active. Deactivate before fooling around

    # Calculate recombination
    if use_rr:
        mesh.calculate_rr_rate(old_mesh, dt)
    #mesh.calculate_tbr_rate(old_mesh, dt) # TODO: Operator split active. Deactivate before fooling around

    return census, mesh

def iterative_solver(census, cutoff, mesh, tol, max_it, seed):
    # Define error metrics and max iterations
    it = 0
    err = np.ones((3, ))*n
    err_per_it = np.zeros((3, max_it))
    Z_bar_five_it = np.zeros((mesh.cells_per_dim[0], 5))
    flux_five_it = np.zeros((mesh.cells_per_dim[0], 5))
    T_five_it = np.zeros((mesh.cells_per_dim[0], 5))
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
        census = List()
        #census = []
        for i in range(len(population)):
            census.append(population[i].copy())
        #census = copy.deepcopy(population)

        # Reset particle tally
        mesh.particle_tally[:] = old_mesh.particle_tally[:]

        # Set the random number seed for generating new particles
        rng = np.random.default_rng(seed)

        time1_recomb = time.time()
        # Use previous iteration's temperatures to source particles from recombination
        if use_rr:
            census = mesh.source_particles_recombination(rng, census, dt)
        time2_recomb = time.time()

        if use_b:
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

        x_loc = 0.15
        x_edges = mesh.get_edges_dim(1)
        x_ind = np.searchsorted(x_edges, x_loc)
        start = x_ind*mesh.cells_per_dim[0]
        end = (x_ind + 1)*mesh.cells_per_dim[0]
        if it < 5:
            Z_bar_five_it[:, it] = np.dot(mesh.ni[start:end, :], np.arange(mesh.N_levels + 1))/mesh.atom_density[start:end]
            flux_five_it[:, it] = mesh.flux[start:end]
            T_five_it[:, it] = mesh.Te[start:end]
        
        it += 1

        time2_it = time.time()
        per_iteration[0] += time2_it - time1_it
        per_iteration[1] += 1

    if it == max_it and be_verbose:
        print("Max iterations reached at t={0}".format(t))
        print("Ni rel err: {:g}".format(err[0]/tol[0]))
        print("Flux rel err: {:g}".format(err[1]/tol[1]))
        print("T rel err {:g} \n".format(err[2]/tol[2]))

        #plt.figure(55)
        #plt.plot(np.arange(max_it), err_per_level.T)
        #plt.xlabel("N iterations")
        #plt.ylabel("Err")
        #plt.legend([str(level) for level in range(mesh.N_levels + 1)])
        #plt.title("Error per level")
#
        #for i in range(3):
        #    plt.figure(50 + i)
        #    plt.subplot(2, 1, 2)
        #    plt.semilogy(np.arange(max_it), err_per_it[i, :], label='t=' + str(dt*t))
        #    plt.xlabel("N iterations")
        #    plt.ylabel("Err")
#
        #symbols = ['o', 'v', '^', '<', '>']
#
        #for i in range(5):
        #    plt.figure(51)
        #    plt.subplot(2, 1, 1)
        #    plt.plot(mesh.get_centers_dim(0), flux_five_it[:, i], label=str(i))
        #    plt.title('Flux')
        #    plt.figure(52)
        #    plt.subplot(2, 1, 1)
        #    plt.plot(mesh.get_centers_dim(0), T_five_it[:, i], label=str(i))
        #    plt.title('Temperature')
        #    plt.figure(50)
        #    plt.subplot(2, 1, 1)
        #    plt.plot(mesh.get_centers_dim(0), Z_bar_five_it[:, i], symbols[i], label=str(i))
        #    plt.title(r'$\overline{Z}$')
        #
        #plt.figure(50)
        #plt.legend()
        #plt.figure(51)
        #plt.legend()
        #plt.figure(52)
        #plt.legend()
        #plt.show()

    # TODO: Operator split - perform eii and tbr calculations after converging on photoionization
    if use_eii:
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
type_photon = Particle(0.1, 0.1, 0.5, np.pi, 800.0, 1.0, 0, 0, 0.0)
census = List.empty_list(typeof(type_photon))
#census = []

# Open file to write to
index = 0
state_file = file_base + 'StateData' + str(index) + '.txt'
while os.path.exists(state_file):
    print('File \'' + state_file + '\' already exists, creating ', end='')
    index += 1
    state_file = file_base + 'StateData' + str(index) + '.txt'
    print('file \'' + state_file + '\' instead')
f = open(state_file, 'w')
f.close()

spectrum_file = file_base + 'SpectrumData' + str(index) + '.txt'
f = open(spectrum_file, 'w')
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

    if t == 0 and plot_initial_energy_spectrum:
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

    roulette_pool = np.zeros((mesh.N_cells, ))

    # Roulette particles
    kill_prob = N_max/len(census)

    if kill_prob < 1:
        index = 0

        rad_energy_per_cell_bfr = np.zeros((mesh.N_cells, ))
        rad_energy_per_cell_aft = np.zeros((mesh.N_cells, ))
        
        for particle in census:
            cell_index = mesh.get_index(particle.cell_i, particle.cell_k)
            rad_energy_per_cell_bfr[cell_index] += particle.w*Constants.h*particle.nu

        while index < len(census):
            cell_index = mesh.get_index(census[index].cell_i, census[index].cell_k)
            if rng.random() > kill_prob and mesh.particle_tally[cell_index] > 1:
                dead_particle = census.pop(index)
                roulette_pool[cell_index] += Constants.h*dead_particle.nu*dead_particle.w
                mesh.particle_tally[cell_index] -= 1
            else:
                index += 1

        for particle in census:
            cell_index = mesh.get_index(particle.cell_i, particle.cell_k)
            particle.roulette(roulette_pool, mesh.particle_tally, cell_index)
            rad_energy_per_cell_aft[cell_index] += particle.w*Constants.h*particle.nu

    roulette_pool = np.zeros((mesh.N_cells, ))
    time2 = time.time()
    russian_roulette[0] += time2 - time1
    russian_roulette[1] += 1

    if t*dt >= plot_time:
        x_line = 0.15
        plot.plot_temperatures_lineout(mesh, t*dt, Ts, x_line)
        plot.plot_temperatures_multicell(mesh, t*dt, Ts, 0, 1)
        plot.plot_average_ionization_level_lineout(mesh, t*dt, x_line)
        plot.plot_average_ionization_level_multicell(mesh, t*dt, 0, 1)
        plot.plot_radiation_spectrum(mesh, census, x_line, 0.0035, t*dt, t*dt > t_max - plot_interval)
        plot.plot_radiation_spectrum(mesh, census, x_line, 0.2, t*dt, t*dt > t_max - plot_interval)
        plot.plot_radiation_spectrum(mesh, census, x_line, 0.35, t*dt, t*dt > t_max - plot_interval)
        plot.plot_alpha_lineout(mesh, t*dt, x_line)
        plot.write_state_data(mesh, state_file, t*dt)
        plot.write_spectrum_data(mesh, census, spectrum_file, x_line, 0.0035, t*dt)
        plot.write_spectrum_data(mesh, census, spectrum_file, x_line, 0.2, t*dt)
        plot.write_spectrum_data(mesh, census, spectrum_file, x_line, 0.35, t*dt)
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

plot.plot_ionization_level_lineout(mesh, t_max, 0.0035)
plt.show()

plot.write_state_data(mesh, state_file, np.round(t*dt, 2))
plot.write_spectrum_data(mesh, census, spectrum_file, x_line, 0.0035, t*dt)
plot.write_spectrum_data(mesh, census, spectrum_file, x_line, 0.2, t*dt)
plot.write_spectrum_data(mesh, census, spectrum_file, x_line, 0.35, t*dt)

if be_verbose:
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

if inputs["Outputs"]["Energy Balance"]:
    time = np.linspace(0, t_max, steps)
    plt.plot(time, dEpi_vec, label='$dE_{pi}$')
    plt.plot(time, dErr_vec, label='$dE_{rr}$')
    plt.legend()
    plt.show()