"""
This program implements the main loop for PI Front transport with implicit Monte Carlo.
It computes an effective scattering term and uses this in addition to sourcing particles
from emission. 
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time
import os
import sys
from numba.typed import List
from numba import float64, typeof, jit

import Utils.CrossSectionFunctions as xsf
import Utils.Constants as Constants
from MonteCarlo.Particle import Particle

N_source = 10_000
N_recomb = 100

M = 100
x_edges = [0, 100]
plane_area = x_edges[1] - x_edges[0]
z_edges = np.linspace(0.0, 0.7, M + 1)
z_centers = z_edges[:-1] + np.diff(z_edges)

mat = xsf.Hydrogen()
n0 = 7e20
T = np.ones((M, ))*1e-5
flux = np.zeros((M, ))
ni = np.ones((M, ))*n0
dni = np.zeros((M, ))
dEpi = np.zeros((M, ))
dErr = np.zeros((M, ))

times = np.linspace(0, 1, 1000)
dt = times[1] - times[0]
print_time = 0.2
print_int = 0.2

T_s = 0.1
source_flux = dt*plane_area*2*np.pi*Constants.sigma_SB*T_s**4/(Constants.keV2GJ)    # [Number of particles]

rng = np.random.default_rng()

census = []
cutoff = 1e-8

end = False

for t in times:
    for _ in range(N_source):
        x = 50
        cell_i = 0
        z = 0
        cell_k = 0

        mu = np.sqrt(rng.random())
        phi = 2*np.pi*rng.random()
        xi = rng.random(5)
        nu = xsf.sample_blackbody(T_s, xi)
        w = source_flux/(N_source*Constants.h*nu)

        timestep_dist = Constants.c*dt*rng.random()

        particle = Particle(x, z, mu, phi, nu, w, cell_i, cell_k, timestep_dist)

        census.append(particle)

    dni = np.zeros((M, ))
    dEpi = np.zeros((M, ))
    dErr = np.zeros((M, ))

    index = 0
    while index < len(census):
        # For each particle in the census, move and compute
        particle = census[index]
        outside_bounds = False
        while particle.timestep_dist > 0 and not outside_bounds:
            # Get particle cell locations
            cell_i = particle.cell_i
            cell_k = particle.cell_k

            # Compute cell volume
            cell_vol = z_edges[cell_k + 1] - z_edges[cell_k]

            # Compute material parameters for the cell, such as neutral ions, fleck factor and scattering cross-section
            n1 = n0 - ni[cell_k]                                        # Number density of ions
            # sigma_pi_avg = mat.sigma_pi_bar(T[cell_k])*ni[cell_k]
            sigma_pi = mat.pi_n(Constants.h*particle.nu)*ni[cell_k]     # Total absorption cross section  
            f = mat.flec_fact(T[cell_k], n1, dt)
            #sigma_s = (1 - f)*sigma_pi_avg
            sigma_s = (1 - f)*sigma_pi                                  # Effective scattering cross section

            xi = rng.random()

            # Sample a distance to next collision
            if sigma_s != 0:
                lam_s = -math.log(xi)/sigma_s
            else:
                lam_s = 1.1*Constants.c*dt

            # Compute effective absorption cross-section
            sigma_pi_eff = f*sigma_pi

            if sigma_pi_eff < 0 or sigma_s < 0:
                print(f"Total absorption: {mat.pi_n(Constants.h*particle.nu)*ni[cell_k]}")
                print(f"Effective scattering: {sigma_s}")
                print(f"Fleck factor: {f}")
                print(f"Ni: {n1}")
                print(f"T: {T[cell_k]}")
                print(f"Recomb rate coeff: {mat.rr_n(T[cell_k])}")
                assert 0

            # Advance particle
            s = particle.move([x_edges[cell_i], x_edges[cell_i + 1]], [z_edges[cell_k], z_edges[cell_k + 1]], lam_s)

            # Compute change in flux in cell via implicit capture
            if sigma_pi_eff < 1e-8:
                dflux = s*particle.w/(cell_vol*dt)
            else:
                dflux = 1/(sigma_pi_eff*cell_vol*dt)*particle.w*(1 - math.exp(-sigma_pi_eff*s))

            flux[cell_k] += dflux

            # Reduce particle weight from absorption
            particle.reduce_weight(s, sigma_pi_eff)

            if s == lam_s and Constants.h*particle.nu > mat.Eth:
                # If scattering event occured, change particle direction and energy
                particle.mu = 2*rng.random() - 1
                particle.phi = 2*np.pi*rng.random()
                dEpi += particle.w*(Constants.h*particle.nu - mat.Eth)
                particle.nu = (xsf.sample_maxwellian(T[particle.cell_k], rng) + mat.Eth)/Constants.h
                dErr += particle.w*(Constants.h*particle.nu - mat.Eth)

            # Compute amount of photoionization
            photoi = max(min(ni[cell_k] + dni[cell_k], sigma_pi_eff*dflux*dt), 0.0)

            dni[cell_k] -= photoi
            dEpi[cell_k] += photoi*(Constants.h*particle.nu - mat.Eth)

            if particle.cell_k < 0 or particle.cell_k >= M:
                # Particle has left domain. Stop tracking it over this timestep
                outside_bounds = True

        if particle.w < cutoff or outside_bounds:
            # Remove particle if it has left domain
            census.pop(index) 
        else:           
            index += 1

        # Reset particle distance traveled for next time step
        particle.timestep_dist = Constants.c*dt

    # Source particles from recombination
    for cell_k in range(M):
        cell_i = 0
        x = 50
        n1 = n0 - ni[cell_k]
        f = mat.flec_fact(T[cell_k], n1, dt)
        cell_vol = z_edges[cell_k + 1] - z_edges[cell_k]
        recomb_density = max(min(n0 - (ni[cell_k] + dni[cell_k]), f*mat.rr_n(T[cell_k])*n1**2*dt), 0.0)
        w0 = recomb_density*cell_vol/N_recomb
        #dni[cell_k] += w0
        for _ in range(N_recomb):
            z = rng.random()*(z_edges[cell_k + 1] - z_edges[cell_k]) + z_edges[cell_k]
            mu = 2*rng.random() - 1
            phi = 2*math.pi*rng.random()

            E = xsf.sample_maxwellian(T[cell_k], rng)
            nu = (E + mat.Eth)/Constants.h

            timestep_dist = Constants.c*dt*rng.random()

            if w0 > cutoff:
                dni[cell_k] += recomb_density/N_recomb
                photon = Particle(x, z, mu, phi, nu, w0, cell_i, cell_k, timestep_dist)
                census.append(photon)
                dErr[cell_k] += w0*E

    # if np.any(dni > 0):
    #     cell = np.argmax(dni)
    #     print(f"Maximum change in ion state: {np.max(dni)}")
    #     print(f"Cell: {cell}")
    #     print(f"Current ion state: {ni[cell]}")
    #     print(f"cv: {xsf.cv(T, 2*n0 - ni[cell])}")
    #     assert 0

    ni += dni
    T += (dEpi - dErr)/xsf.cv(T, 2*n0 - ni) #  + 1.5*dni*T

    if np.any(ni < 0):
        print(f"Ni less than 0 at time {t}. Minimum {np.min(ni)}")
        end = True
    if np.any(T < 0):
        print(f"T less than zero at time {t}. Minimum temp {np.min(T)}")
        ind = np.argmin(T)
        print(f"dEpi: {dEpi[ind]} \n dErr: {dErr[ind]} n1: {ni[ind]} \n Cell: {ind}")

        end = True

    if end:
        assert 0

    if t > print_time:
        plt.figure(1)
        plt.plot(z_centers, (n0 - ni)/n0, label=str(np.round(t, 2)) + ' ns')

        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(z_centers, (flux*Constants.keV2GJ/(Constants.a*Constants.c))**0.25, label=str(np.round(t, 2)) + ' ns')
        plt.subplot(2, 1, 2)
        plt.plot(z_centers, T, label=str(np.round(t, 2)) + ' ns')

        print(np.round(print_time, 2))

        print_time += print_int

plt.figure(1)
plt.plot(z_centers, (n0 - ni)/n0, label=str(np.round(t, 2)) + ' ns')
plt.title('Ionization Fraction')
plt.xlabel('z-location')
plt.ylabel('Ion fraction [-]')
plt.legend()
plt.figure(2)
plt.subplot(2, 1, 1)
plt.title('Rad. Temp.')
plt.plot(z_centers, (flux*Constants.keV2GJ/(Constants.a*Constants.c))**0.25, label=str(np.round(t, 2)) + ' ns')
plt.ylabel('Temperature [keV]')
plt.subplot(2, 1, 2)
plt.title('Mat. Temp.')
plt.plot(z_centers, T, label=str(np.round(t, 2)) + ' ns')
plt.xlabel('z-location')
plt.ylabel('Temperature [keV]')
plt.show()