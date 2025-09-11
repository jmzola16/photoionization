# This file contains a class which contains all data related to a mesh for a 
# photoionization simulation, such as the ionization level and electron
# temperature in each cell
import numpy as np
import sys
sys.path.append("..")
import Constants
import CrossSectionFunctions as xsf
from Particle import Particle

class Mesh:
    def __init__(self, cell_edges, N_levels, cell_mats, atom_density, recombination_N_per_cell):
        # Initialize cell information
        self.N_levels = N_levels
        self.N_cells = len(cell_edges) - 1
        self.cell_edges = cell_edges
        self.cell_widths = np.diff(cell_edges)
        self.cell_centers = cell_edges[:-1] + self.cell_widths

        # Initialize temperature profile
        self.Te = 1e-5*np.ones((self.N_cells, ))

        # Initialize flux
        self.flux = np.zeros((self.N_cells, ))

        # Initialize ionization state
        self.ni = np.zeros((self.N_cells, N_levels + 1))
        self.dni = np.zeros((self.N_cells, N_levels + 1))

        if hasattr(atom_density, "__len__"):
            self.ni[:, 0] = atom_density
        else:
            self.ni[:, 0] = atom_density*np.ones((self.N_cells, ))

        self.ne = np.matmul(self.ni, np.arange(N_levels + 1))

        # Initialize material in each cell
        if hasattr(cell_mats, "__len__"):
            self.cell_mats = cell_mats
        else:
            self.cell_mats = []
            # cell_mats = List()
            for i in range(self.N_cells):
                self.cell_mats.append(cell_mats)

        # Initialize number of particles in each cell
        self.particle_tally = np.zeros((self.N_cells, ), dtype=int)

        # Initialize photoionization and recombination data storage
        self.recombination_N_per_cell = recombination_N_per_cell
        self.recomb_rate = np.zeros((self.N_cells, self.N_levels - 1))
        self.photoi_rate = np.zeros((self.N_cells, self.N_levels - 1))

        # Initialize energy change variables
        self.dEpi = np.zeros((self.N_cells, ))
        self.dErr = np.zeros((self.N_cells, ))

    def absorb_particle(self, photon, ni_old, dt):
        # Store current cell locally, because it will change when particle is moved
        cell = photon.cell

        # Calculate the photoionization cross-section for each level
        Gamma = np.zeros((self.N_levels - 1, ))
        for level in range(self.N_levels - 1):
            Gamma[level] = self.cell_mats[cell]*self.ni[cell, level]
        Gamma_tot = np.sum(Gamma)

        # Move the particle, updating the cell tally in the process
        self.particle_tally[cell] -= 1

        s = photon.move(self.cell_edges[cell:(cell + 2)])

        if photon.cell >= 0 and photon.cell < self.N_cells:
            self.particle_tally[photon.cell] += 1

        # Calculate the flux from implicit capture of this photon
        if Gamma_tot < 1e-8:
            dflux = s*photon.w/(self.cell_widths[cell]*dt)*photon.w*(1 - np.exp(-Gamma_tot*s))
        else:
            dflux = 1/(Gamma_tot*self.cell_widths[cell]*dt)

        self.flux[photon.cell] += dflux
        self.energy_density[photon.cell] += dflux*Constants.h*photon.nu/Constants.c

        for level in range(self.N_levels):
            # Calculate the number of photoionized particles for this ionization state
            # If this would photoionize more than remaining particles, reduce to zero instead
            photoi = min(ni_old[cell, level] + self.dni[cell, level], Gamma[level]*dflux*dt)

            self.dni[cell, level] -= photoi
            self.dni[cell, level + 1] += photoi
            self.photoi_rate[cell, level] += photoi
            self.dEpi[cell] += photoi*(Constants.h*photon.nu - self.cell_mats[cell].Eth[level])

        photon.reduce_weight(s, Gamma_tot)

        return 0

    def reset_time_step(self):
        self.photoi_rate = np.zeros((self.N_cells, self.N_levels - 1))
        self.flux = np.zeros((self.N_cells, ))
        self.Epi = np.zeros((self.N_cells, ))
        self.Err = np.zeros((self.N_cells, ))

        return 0

    def alpha(self):
        alpha = np.zeros((self.N_cells, self.N_levels - 1))
        
        for level in range(self.N_levels):
            alpha[:, level] = self.recomb_rate[:, level]/self.photoi_rate[:, level]
        
        return alpha
    
    def calculate_recombination_rate(self, ni_old):
        # This function calculates the rate of recombination in a photoionizing medium

        tot_recomb_rate = np.zeros((self.N_cells, ))
        N_recomb = np.zeros((self.N_cells, self.N_levels), dtype=int)
        for cell in range(self.N_cells):
            recomb_rate_per_level = np.zeros((self.N_levels, ))
            for level in range(self.N_levels):
                recomb_rate_per_level[level] = min(self.cell_mats[cell].rr_n(self.Te[cell], level + 1)*self.ne[cell]*self.ni[cell, level], ni_old[cell, level] + self.dni[cell, level])
                tot_recomb_rate[cell] += recomb_rate_per_level[level]

            self.recomb_rate[cell, :] = recomb_rate_per_level

            self.dni[cell, :-1] += self.recomb_rate[cell, :]
            self.dni[cell, 1:] -= self.recomb_rate[cell, :]
            self.dErr[cell] += np.sum(self.recomb_rate[cell, :]*(self.cell_mat[cell].Eth + 1.5*self.Te[cell]))

            if tot_recomb_rate[cell] > 1e-4:
                N_recomb[cell, :] = int(recomb_rate_per_level*self.recombination_N_per_cell/tot_recomb_rate[cell])

        return self.recomb_rate, N_recomb
    
    def source_particles_recombination(self, rng, census, N_recomb, dt):
        # This function adds particles to a census after sourcing them from recombination
        for cell in range(self.N_cells):
            for level in range(self.N_levels):
                for particle in range(N_recomb[cell, level]):
                    pos_in_cell = self.cell_edges[cell] + rng.random()*self.cell_widths
                    mu = rng.random()*2 - 1
                    wgt = self.recomb_rate[cell, level]*dt/(N_recomb[cell, level])
                    start_time = dt*rng.random()
                    xi = rng.random()

                    nu = (self.cell_mats[cell].Eth[level] + xsf.sample_maxwellian(self.Te[cell], xi))/Constants.h
                    photon = Particle(pos_in_cell, mu, nu, wgt, cell, (dt - start_time)/dt*Constants.c*dt)

                    census.append(photon)
                    self.particle_tally[cell] += 1