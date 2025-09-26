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

        # Initialize flux and energy density
        self.flux = np.zeros((self.N_cells, ))
        self.energy_density = np.zeros((self.N_cells, ))

        # Initialize ionization state
        self.ni = np.zeros((self.N_cells, N_levels + 1))
        self.dni = np.zeros((self.N_cells, N_levels + 1))

        if hasattr(atom_density, "__len__"):
            self.ni[:, 0] = atom_density
        else:
            self.ni[:, 0] = atom_density*np.ones((self.N_cells, ))

        self.atom_density = self.ni[:, 0]

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
        self.recomb_rate = np.zeros((self.N_cells, self.N_levels))
        self.N_recomb = np.zeros((self.N_cells, self.N_levels), dtype=int)
        self.photoi_rate = np.zeros((self.N_cells, self.N_levels))

        # Initialize energy change variables
        self.dEpi = np.zeros((self.N_cells, ))
        self.dErr = np.zeros((self.N_cells, ))

    def absorb_particle(self, photon, old_mesh, dt):
        # Store current cell locally, because it will change when particle is moved
        cell = photon.cell

        # Calculate the photoionization cross-section for each level
        Gamma = np.zeros((self.N_levels, ))
        for level in range(self.N_levels):
            Gamma[level] = self.cell_mats[cell].pi_n(Constants.h*photon.nu, level)*self.ni[cell, level]
        Gamma_tot = np.sum(Gamma)

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

        self.flux[cell] += dflux
        self.energy_density[cell] += dflux*Constants.h*photon.nu/Constants.c

        for level in range(self.N_levels):
            # Calculate the number of photoionized particles for this ionization state
            # If this would photoionize more than remaining particles, reduce to zero instead
            photoi = min(old_mesh.ni[cell, level] + self.dni[cell, level], Gamma[level]*dflux*dt)

            self.dni[cell, level] -= photoi
            self.dni[cell, level + 1] += photoi
            self.photoi_rate[cell, level] += photoi
            self.dEpi[cell] += photoi*(Constants.h*photon.nu - self.cell_mats[cell].Eth[level])

        photon.reduce_weight(s, Gamma_tot)

        return 0
    
    def calculate_recombination_rate(self, old_mesh, dt):
        # This function calculates the rate of recombination in a photoionizing medium
        tot_recomb_rate = np.zeros((self.N_cells, ))
        for cell in range(self.N_cells):
            recomb_rate_per_level = np.zeros((self.N_levels, ))
            for level in range(self.N_levels):
                recomb_rate_per_level[level] = min(self.cell_mats[cell].rr_n(self.Te[cell], level + 1)*self.ne[cell]*self.ni[cell, level + 1]*dt, old_mesh.ni[cell, level + 1] + self.dni[cell, level + 1])
                tot_recomb_rate[cell] += recomb_rate_per_level[level]

            self.recomb_rate[cell, :] = recomb_rate_per_level

            self.dni[cell, :-1] += self.recomb_rate[cell, :]
            self.dni[cell, 1:] -= self.recomb_rate[cell, :]
            self.dErr[cell] += np.sum(self.recomb_rate[cell, :]*(self.cell_mats[cell].Eth + 1.5*self.Te[cell]))

            if tot_recomb_rate[cell] > 1e-4:
                self.N_recomb[cell, :] = np.ceil(recomb_rate_per_level*self.recombination_N_per_cell/tot_recomb_rate[cell]).astype(int)

        return 0
    
    def source_particles_recombination(self, rng, census, dt):
        # This function adds particles to a census after sourcing them from recombination
        cum_dErr = np.zeros((self.N_cells, ))
        for cell in range(self.N_cells):
            for level in range(self.N_levels):
                for particle in range(self.N_recomb[cell, level]):
                    pos_in_cell = self.cell_edges[cell] + rng.random()*self.cell_widths[cell]
                    mu = rng.random()*2 - 1
                    wgt = self.recomb_rate[cell, level]/(self.N_recomb[cell, level])
                    start_time = dt*rng.random()
                    xi = rng.random()

                    cum_dErr[cell] += wgt*(self.cell_mats[cell].Eth[level] + xsf.sample_maxwellian(self.Te[cell], xi))

                    nu = (self.cell_mats[cell].Eth[level] + xsf.sample_maxwellian(self.Te[cell], xi))/Constants.h
                    photon = Particle(pos_in_cell, mu, nu, wgt, cell, (dt - start_time)/dt*Constants.c*dt)

                    census.append(photon)
                    self.particle_tally[cell] += 1

        return census

    def alpha(self):
        alpha = np.zeros((self.N_cells, self.N_levels))
        
        for level in range(self.N_levels):
            alpha[:, level] = self.recomb_rate[:, level]/self.photoi_rate[:, level]
        
        return alpha
    
    def reset_iteration(self):
        # Reset atomic kinetics rates for this iteration
        self.photoi_rate = np.zeros((self.N_cells, self.N_levels))
        self.recomb_rate = np.zeros((self.N_cells, self.N_levels))
        self.N_recomb = np.zeros((self.N_cells, self.N_levels), dtype=int)
        self.dni = np.zeros((self.N_cells, self.N_levels + 1))

        # Reset radiation transport variables for this iteration
        self.flux = np.zeros((self.N_cells, ))
        self.energy_density = np.zeros((self.N_cells, ))

        # Reset temperature update variables for this iteration
        self.dEpi = np.zeros((self.N_cells, ))
        self.dErr = np.zeros((self.N_cells, ))

        return 0
    
    def copy(self):
        copied_mesh = Mesh(self.cell_edges, self.N_levels, self.cell_mats, self.atom_density, self.recombination_N_per_cell)
        copied_mesh.ni[:] = self.ni[:]
        copied_mesh.ne[:] = self.ne[:]
        copied_mesh.Te[:] = self.Te[:]
        copied_mesh.particle_tally[:] = self.particle_tally[:]

        return copied_mesh
    
    def update_state(self, old_mesh, dt, use_dEsp=False):
        self.ni = old_mesh.ni + self.dni
        #dne = np.matmul(self.dni, np.arange(self.N_levels + 1))
        self.ne = np.matmul(self.ni, np.arange(self.N_levels + 1))
        dne = self.ne - old_mesh.ne

        dEsp = np.zeros((self.N_cells, ))
        if use_dEsp:
            for cell in range(self.N_cells):
                dEsp[cell] = self.cell_mats[cell].E_spectral(self.Te[cell], self.ni[cell, :])*dt

        dT = (self.dEpi - self.dErr - dEsp - 1.5*self.Te*dne)/xsf.cv(self.Te, self.atom_density + self.ne)
        self.Te = old_mesh.Te + dT

    def plot_ionization_level(self, t):
        plt.figure()
        for level in range(self.N_levels):
            plt.plot(self.cell_centers, self.ni[:, level]/self.atom_density)
        plt.xlabel('z-location [cm]')
        plt.ylabel('Ion Fraction [-]')
        plt.legend(['n' + str(i) for i in range(self.N_levels + 1)])
        plt.title('Ion fraction at t=' + str(np.round(t, 2)) + ' ns')

    def plot_average_ionization_level(self, t):
        plt.figure(11)
        z_bar = np.zeros((self.N_cells, ))
        for i in range(self.N_levels + 1):
            z_bar += i*self.ni[:, i]/self.atom_density

        plt.plot(self.cell_centers, z_bar, label='t='+str(np.round(t, 2)))
        plt.xlabel('z-location [cm]')
        plt.ylabel(r'$ \overline{Z} $')
    
    def plot_temperatures(self, t):
        plt.figure(22)
        plt.subplot(2, 1, 2)
        plt.plot(self.cell_centers, self.Te, label='t='+str(np.round(t, 2)))
        plt.xlabel('z-location [cm]')
        plt.ylabel('Material temperature [keV]')
        plt.subplot(2, 1, 1)
        plt.plot(self.cell_centers, (self.energy_density*Constants.keV2GJ/Constants.a)**0.25, label='t='+str(np.round(t, 2)))
        plt.ylabel('Radiation temperature [keV]')

    def plot_alpha(self, t):
        plt.figure()
        plt.plot(self.cell_centers, self.alpha())
        plt.title(r'$ \alpha $ at t=' + str(round(t, 2)))
        plt.xlabel('z-location [cm]')
        plt.ylabel(r'$ \alpha $ [-]')
        plt.legend(['n' + str(i) for i in range(self.N_levels + 1)])

    def write_data(self, file, t):
        f = open(file, 'a')
        f.write('{:4.2g}'.format(t))
        f.write('\n\n')
        form = ''
        for i in range(self.N_cells):
            form += '{:e}, '.format(self.energy_density[i])
        f.write(form)
        f.write('\n\n')

        form = ''
        for i in range(self.N_cells):
            form += '{:e}, '.format(self.T[i])
        f.write(form)
        f.write('\n\n')

        for i in range(self.N_cells):
            form = ''
            for j in range(self.N_levels):
                form += '{:e}, '.format(self.ni[i, j])

            f.write(form + '\n')
    
        f.write('\n')
        f.close()    