# This file contains a class which contains all data related to a mesh for a 
# photoionization simulation, such as the ionization level and electron
# temperature in each cell
import numpy as np
import sys
sys.path.append("..")
import Constants
import CrossSectionFunctions as xsf
from Particle import Particle
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import int32, float64, typeof, types
from numba.typed import List
import math

type_material = typeof(xsf.Nitrogen())

spec = [
    ('N_levels', int32),
    ('N_cells', int32),
    ('cell_edges', float64[:]),
    ('cell_widths', float64[:]),
    ('cell_centers', float64[:]),
    ('Te', float64[:]),
    ('dT', float64[:]),
    ('N_groups', int32),
    ('energy_group_edges', float64[:]),
    ('multigroup_flux', float64[:, :]),
    ('flux', float64[:]),
    ('energy_density', float64[:]),
    ('ni', float64[:, :]),
    ('dni', float64[:, :]),
    ('atom_density', float64[:]),
    ('ne', float64[:]),
    ('cell_mats', types.ListType(type_material)),
    ('particle_tally', int32[:]),
    ('recombination_N_per_cell', int32),
    ('bremsstrahlung_N', int32),
    ('recomb_rate', float64[:, :]),
    ('N_recomb', int32[:, :]),
    ('photoi_rate', float64[:, :]),
    ('dEpi', float64[:]),
    ('dEib', float64[:]),
    ('dEb', float64[:]),
    ('dErr', float64[:])
]

@jitclass(spec)
class Mesh:
    def __init__(self, cell_edges, N_levels, cell_mats, atom_density, recombination_N_per_cell, bremsstrahlung_N):
        # Initialize cell information
        self.N_levels = N_levels
        self.N_cells = len(cell_edges) - 1
        self.cell_edges = cell_edges
        self.cell_widths = np.diff(cell_edges)
        self.cell_centers = cell_edges[:-1] + self.cell_widths

        # Initialize temperature profile
        self.Te = 1e-5*np.ones((self.N_cells, ))
        self.dT = np.zeros((self.N_cells, ))

        # Initialize flux, energy groups, and energy density
        self.N_groups = 100
        self.energy_group_edges = np.zeros((self.N_groups + 1, ))
        self.energy_group_edges[:5] = np.logspace(-9, -4, 5)
        self.energy_group_edges[5:] = np.linspace(1e-3, 1, self.N_groups - 4)
        self.multigroup_flux = np.zeros((self.N_cells, self.N_groups))
        self.flux = np.zeros((self.N_cells, ))
        self.energy_density = np.zeros((self.N_cells, ))

        # Initialize ionization state
        self.ni = np.zeros((self.N_cells, N_levels + 1))
        self.dni = np.zeros((self.N_cells, N_levels + 1))

        if hasattr(atom_density, "__len__"):
            self.ni[:, 0] = atom_density
        else:
            self.ni[:, 0] = atom_density*np.ones((self.N_cells, ))

        self.atom_density = self.ni[:, 0].copy()

        self.ne = np.dot(self.ni, np.arange(N_levels + 1, dtype=float64))

        # Initialize material in each cell
        if hasattr(cell_mats, "__len__"):
            self.cell_mats = cell_mats
        else:
            #self.cell_mats = []
            self.cell_mats = List.empty_list(type_material)
            for i in range(self.N_cells):
                self.cell_mats.append(cell_mats)

        # Initialize number of particles in each cell
        self.particle_tally = np.zeros((self.N_cells, ), dtype=int32)

        # Initialize photoionization and recombination data storage
        self.recombination_N_per_cell = recombination_N_per_cell
        self.bremsstrahlung_N = bremsstrahlung_N
        self.recomb_rate = np.zeros((self.N_cells, self.N_levels))
        self.N_recomb = np.zeros((self.N_cells, self.N_levels), dtype=int32)
        self.photoi_rate = np.zeros((self.N_cells, self.N_levels))

        # Initialize energy change variables
        self.dEpi = np.zeros((self.N_cells, ))
        self.dEib = np.zeros((self.N_cells, ))
        self.dEb = np.zeros((self.N_cells, ))
        self.dErr = np.zeros((self.N_cells, ))

    def absorb_particle(self, photon, old_mesh, use_b, dt):
        # Store current cell locally, because it will change when particle is moved
        cell = photon.cell

        # Calculate the photoionization cross-section for each level
        Gamma_pi = np.zeros((self.N_levels, ))
        Gamma_ib = np.zeros((self.N_levels, ))
        for level in range(self.N_levels):
            Gamma_pi[level] = self.cell_mats[cell].pi_n(Constants.h*photon.nu, level)*self.ni[cell, level]
            if use_b:
                Gamma_ib[level] = self.cell_mats[cell].ibrem_xs(self, cell, photon.nu, 2)*self.ni[cell, level]
        Gamma_ib[0] = 0 # TODO: Check if this is correct
        Gamma_tot = np.sum(Gamma_pi) + np.sum(Gamma_ib)

        # Move the particle, updating the cell tally in the process
        self.particle_tally[cell] -= 1

        s = photon.move(self.cell_edges[cell:(cell + 2)])

        if photon.cell >= 0 and photon.cell < self.N_cells:
            self.particle_tally[photon.cell] += 1

        # Calculate the flux from implicit capture of this photon
        if Gamma_tot < 1e-8:
            dflux = s*photon.w/(self.cell_widths[cell]*dt)
        else:
            dflux = 1/(Gamma_tot*self.cell_widths[cell]*dt)*photon.w*(1 - np.exp(-Gamma_tot*s))

        group = self.find_group(Constants.h*photon.nu)
        self.multigroup_flux[cell, group] += dflux*Constants.h*photon.nu/Constants.c
        self.flux[cell] += dflux
        self.energy_density[cell] += dflux*Constants.h*photon.nu/Constants.c

        for level in range(self.N_levels):
            # Calculate the number of photoionized particles for this ionization state
            # If this would photoionize more than remaining particles, reduce to zero instead
            photoi = min(old_mesh.ni[cell, level] + self.dni[cell, level], Gamma_pi[level]*dflux*dt)

            self.dni[cell, level] -= photoi
            self.dni[cell, level + 1] += photoi
            self.photoi_rate[cell, level] += photoi
            self.dEpi[cell] += photoi*(Constants.h*photon.nu - self.cell_mats[cell].Eth[level])
            self.dEib[cell] += Gamma_ib[level]*dflux*dt*Constants.h*photon.nu

        photon.reduce_weight(s, Gamma_tot)

        return 0
    
    def calculate_eii(self, old_mesh, dt):
        # This function calculates the ionization from electron impact
        for cell in range(self.N_cells):
            for level in range(self.N_levels):
                eii_rate = min(self.ne[cell]*self.ni[cell, level]*self.cell_mats[cell].sigma_n(self.Te[cell], level)*dt, old_mesh.ni[cell, level] + self.dni[cell, level])

                self.dni[cell, level] -= eii_rate
                self.dni[cell, level + 1] += eii_rate

        return 0
    
    def calculate_rr_rate(self, old_mesh, dt):
        # This function calculates the rate of recombination in a photoionizing medium
        tot_recomb_rate = np.zeros((self.N_cells, ))
        for cell in range(self.N_cells):
            recomb_rate_per_level = np.zeros((self.N_levels, ))
            for level in range(self.N_levels):
                recomb_rate_per_level[level] = min(self.cell_mats[cell].rr_n(self.Te[cell], level + 1)*self.ne[cell]*self.ni[cell, level + 1]*dt, old_mesh.ni[cell, level + 1] + self.dni[cell, level + 1])
                tot_recomb_rate[cell] += recomb_rate_per_level[level]

            self.recomb_rate[cell, :] = recomb_rate_per_level.copy()

            self.dni[cell, :-1] += self.recomb_rate[cell, :]
            self.dni[cell, 1:] -= self.recomb_rate[cell, :]
            self.dErr[cell] += np.sum(self.recomb_rate[cell, :]*(1.5*self.Te[cell])) # self.cell_mats[cell].Eth + 

            if tot_recomb_rate[cell] > 1e-4:
                self.N_recomb[cell, :] = np.ceil(recomb_rate_per_level*self.recombination_N_per_cell/tot_recomb_rate[cell]).astype(int32)

        return 0
    
    def calculate_tbr_rate(self, old_mesh, dt):
        for cell in range(self.N_cells):
            for level in range(self.N_levels):
                if self.ne[cell] <= 0 or self.ni[cell, level + 1] <= 0:
                    log_space_tbr = 1
                    tbr_rate = 0
                else:
                    #log_space_tbr = np.log(self.cell_mats[cell].tbr_n(self.Te[cell], level + 1)) + 2*np.log(self.ne[cell]) + np.log(self.ni[cell, level + 1])
                    log_space_tbr = self.cell_mats[cell].tbr_n(self.Te[cell], level + 1) + 2*np.log(self.ne[cell]) + np.log(self.ni[cell, level + 1])
                    tbr_rate = min(np.exp(log_space_tbr)*dt, old_mesh.ni[cell, level + 1] + self.dni[cell, level + 1])
                #tbr_rate = min(self.cell_mats[cell].tbr_n(self.Te[cell], level + 1)*self.ne[cell]**2*self.ni[cell, level + 1]*dt, old_mesh.ni[cell, level + 1] + self.dni[cell, level + 1])

                if tbr_rate != tbr_rate:
                    print("Tbr rate Nan")
                    print("Log space tbr rate: ")
                    print(log_space_tbr)
                    print("Ne: ")
                    print(self.ne[cell])
                    print("Ni: ")
                    print(self.ni[cell, level + 1])
                    print("TBR Rate [cm^6 ns^-1]: ")
                    print(self.cell_mats[cell].tbr_n(self.Te[cell], level + 1))
                    print("TBR Rate [cm^-3]: ")
                    print(np.exp(log_space_tbr)*dt)
                    print("Remaining ion density: ")
                    print(old_mesh.ni[cell, level + 1] + self.dni[cell, level + 1])

                    assert 0


                self.dni[cell, level] += tbr_rate
                self.dni[cell, level + 1] -= tbr_rate

        return 0
    
    def source_particles_recombination(self, rng, census, dt):
        # This function adds particles to a census after sourcing them from recombination
        cum_dErr = np.zeros((self.N_cells, ))
        for cell in range(self.N_cells):
            for level in range(self.N_levels):
                for particle in range(self.N_recomb[cell, level]):
                    pos_in_cell = self.cell_edges[cell] + rng.random()*self.cell_widths[cell]
                    mu = rng.random()*2 - 1
                    wgt = self.recomb_rate[cell, level]*self.cell_widths[cell]/(self.N_recomb[cell, level])
                    start_time = dt*rng.random()
                    xi = rng.random()

                    cum_dErr[cell] += wgt*(self.cell_mats[cell].Eth[level] + xsf.sample_maxwellian(self.Te[cell], xi))

                    nu = (self.cell_mats[cell].Eth[level] + xsf.sample_maxwellian(self.Te[cell], xi))/Constants.h # self.cell_mats[cell].Eth[level] + 
                    photon = Particle(pos_in_cell, mu, nu, wgt, cell, (dt - start_time)/dt*Constants.c*dt)

                    census.append(photon)
                    self.particle_tally[cell] += 1

        return census

################### NEW - BREMSSTRAHLUNG SOURCE ##################################################

    def source_particles_bremsstrahlung(self, rng, census, dt):
        for cell in range(self.N_cells):
            unadj_wgt = Constants.sigma_SB*self.Te[cell]**4/(Constants.keV2GJ)
            sum_brems_xs = np.zeros((self.N_cells, ))
            for particle in range(self.bremsstrahlung_N):
                pos_in_cell = self.cell_edges[cell] + rng.random()*self.cell_widths[cell]
                mu = rng.random()*2 - 1
                start_time = dt*rng.random()
                xi = rng.random(5)

                nu = xsf.sample_blackbody(self.Te[cell], xi)

                sigma_brem_tot = self.cell_mats[cell].ibrem_xs(self, cell, nu, 2)*np.sum(self.ni[cell, 1:])

                #wgt = self.cell_mats[cell].ibrem_xs(self, cell, nu, 2)*self.ni[cell, level]*xsf.blackbody(nu, self.Te[cell])[0]*dt*self.cell_widths[cell]*4*np.pi/(self.bremsstrahlung_N*Constants.h)
                #wgt = self.cell_mats[cell].ibrem_xs(self, cell, nu, 2)*self.ni[cell, level]*unadj_wgt*self.cell_widths[cell]/(self.bremsstrahlung_N*self.N_levels*Constants.h*nu)
                #conv_const = 1
                conv_const = Constants.c*dt*4*np.pi/(Constants.h*nu)
                wgt = sigma_brem_tot*unadj_wgt/(self.bremsstrahlung_N)
                sum_brems_xs[cell] += wgt

                photon = Particle(pos_in_cell, mu, nu, conv_const*wgt, cell, (dt - start_time)/dt*Constants.c*dt)

                census.append(photon)
                self.particle_tally[cell] += 1
                
            for index in range(self.bremsstrahlung_N):
                #census[-index - 1].w /= sum_brems_xs[cell]
                self.dEb[cell] += Constants.h*census[-index - 1].nu*census[-index - 1].w

        return census #, sum_brems_xs

#################################################################################################

    def alpha(self):
        alpha = np.zeros((self.N_cells, self.N_levels))
        
        for level in range(self.N_levels):
            alpha[:, level] = self.recomb_rate[:, level]/self.photoi_rate[:, level]
        
        return alpha
    
    def find_group(self, energy):
        if energy > self.energy_group_edges[-1]:
            return self.N_groups - 1
        elif energy < self.energy_group_edges[0]:
            return 0
        else:
            return np.searchsorted(self.energy_group_edges, energy) - 1

    def reset_iteration(self):
        # Reset atomic kinetics rates for this iteration
        self.photoi_rate = np.zeros((self.N_cells, self.N_levels))
        self.recomb_rate = np.zeros((self.N_cells, self.N_levels))
        self.N_recomb = np.zeros((self.N_cells, self.N_levels), dtype=int32)
        self.dni = np.zeros((self.N_cells, self.N_levels + 1))

        # Reset radiation transport variables for this iteration
        self.multigroup_flux = np.zeros((self.N_cells, self.N_groups))
        self.flux = np.zeros((self.N_cells, ))
        self.energy_density = np.zeros((self.N_cells, ))

        # Reset temperature update variables for this iteration
        self.dEpi = np.zeros((self.N_cells, ))
        self.dErr = np.zeros((self.N_cells, ))
        self.dEib = np.zeros((self.N_cells, ))
        self.dEb = np.zeros((self.N_cells, ))

        return 0
    
    def copy(self):
        copied_mesh = Mesh(np.zeros((self.N_cells + 1, )), self.N_levels, self.cell_mats[0], self.atom_density[0], np.zeros((self.N_cells, ), dtype=int32), self.bremsstrahlung_N)
        copied_mesh.ni = self.ni.copy()
        copied_mesh.flux = self.flux.copy()
        copied_mesh.ne = self.ne.copy()
        copied_mesh.Te = self.Te.copy()
        copied_mesh.particle_tally = self.particle_tally.copy()
        copied_mesh.dni = self.dni.copy()
        copied_mesh.dT = self.dT.copy()

        return copied_mesh
    
    def update_state(self, old_mesh, mesh_previt, dt, lam, use_dEsp):
        ni_update = mesh_previt.dni + lam*(self.dni - mesh_previt.dni)
        self.ni = np.maximum(0.0, old_mesh.ni + ni_update)
        self.ne = np.dot(self.ni, np.arange(self.N_levels + 1).astype(float64))
        dne = self.ne - old_mesh.ne

        # TODO: Change cv according to \frac{\partial e}{\partial t} = \frac{\partial e}{partial T}*\frac{\partial T}{\partial t}

        self.dT = (self.dEpi + self.dEib - self.dEb - self.dErr - 1.5*self.Te*dne)/xsf.cv(self.Te, self.atom_density + self.ne)
        T_update = mesh_previt.dT + lam*(self.dT - mesh_previt.dT)
        self.Te = old_mesh.Te + T_update 