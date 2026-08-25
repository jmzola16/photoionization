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
    ('dim', int32),
    ('N_cells', int32),
    ('cells_per_dim', int32[:]),
    ('cell_edges', float64[:]),
    ('cell_widths', float64[:]),
    ('cell_centers', float64[:]),
    ('bcs', int32[:]),
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
    ('photoi_absorb_tot', float64[:]),
    ('dEpi', float64[:]),
    ('dEib', float64[:]),
    ('dEb', float64[:]),
    ('dErr', float64[:]),
    ('dEeii', float64[:]),
    ('dEtbr', float64[:])
]

@jitclass(spec)
class Mesh:
    def __init__(self, dim, cell_edges, N_levels, cells_per_dim, cell_mats, atom_density, recombination_N_per_cell, bremsstrahlung_N, bcs):
        # Initialize cell information
        self.dim = dim
        self.N_levels = N_levels
        self.cells_per_dim = cells_per_dim
        self.N_cells = np.prod(cells_per_dim)
        self.cell_edges = cell_edges
        self.cell_widths = np.zeros((np.sum(self.cells_per_dim), ))
        self.cell_centers = np.zeros((np.sum(self.cells_per_dim), ))
        start = 0
        end = self.cells_per_dim[0]
        for i in range(dim):
            temp_slice = np.ascontiguousarray(self.cell_edges[(start + i):(end + i + 1)])
            self.cell_widths[start:end] = np.diff(temp_slice)
            self.cell_centers[start:end] = self.cell_edges[(start + i):(end + i)] + self.cell_widths[start:end]/2
            start += end
            if i < dim - 1:
                end += self.cells_per_dim[i + 1]
        self.bcs = bcs

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

        if len(atom_density) > 1:
            self.ni[:, 0] = atom_density
        else:
            self.ni[:, 0] = atom_density[0]*np.ones((self.N_cells, ))

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
        self.photoi_absorb_tot = np.zeros((self.N_cells, ))
        self.dEpi = np.zeros((self.N_cells, ))
        self.dEib = np.zeros((self.N_cells, ))
        self.dEb = np.zeros((self.N_cells, ))
        self.dErr = np.zeros((self.N_cells, ))
        self.dEeii = np.zeros((self.N_cells, ))
        self.dEtbr = np.zeros((self.N_cells, ))

    def absorb_particle(self, photon, old_mesh, use_b, dt):
        # Store current cell locally, because it will change when particle is moved
        cell_i = photon.cell_i
        cell_k = photon.cell_k

        index = self.get_index(cell_i, cell_k)

        # Calculate the photoionization cross-section for each level
        Gamma_pi = np.zeros((self.N_levels, ))
        Gamma_ib = np.zeros((self.N_levels, ))
        if self.atom_density[index] > 1e12:
            # Inverse bremsstrahlung absorption is offset from photoionization absorption, since only
            # ions can absorb via ib
            if use_b:
                Gamma_ib = self.cell_mats[index].ibrem_xs(self, index, photon.nu, 2)*self.ni[index, 1:]
            for level in range(self.N_levels):
                if abs(old_mesh.ni[index, level] + self.dni[index, level]) > max(1e-9*max(abs(old_mesh.ni[index, level]), abs(self.dni[index, level])), 1e3):
                    Gamma_pi[level] = self.cell_mats[index].pi_n(Constants.h*photon.nu, level)*self.ni[index, level]
            #    if use_b:
            #        Gamma_ib[level] = self.cell_mats[index].ibrem_xs(self, index, photon.nu, 2)*self.ni[index, level]
            # Gamma_ib[0] = 0 # TODO: Check if this is correct
        Gamma_tot = np.sum(Gamma_pi) + np.sum(Gamma_ib)

        # Calculate the volume of the cell
        cell_vol = self.cell_widths[cell_k]
        lin_index = 0
        for i in range(self.dim - 1):
            lin_index += self.cells_per_dim[i]
            cell_vol *= self.cell_widths[lin_index + cell_i]
        # cell_vol = self.cell_widths[cell_i]*self.cell_widths[self.cells_per_dim[0] + cell_k]

        photoi_dist = photon.timestep_dist
        for level in range(self.N_levels):
            if Gamma_pi[level] > 0 and abs(old_mesh.ni[index, level] + self.dni[index, level]) > max(1e-9*max(abs(old_mesh.ni[index, level]), abs(self.dni[index, level])), 1e4):
                photoi_dist = min(photoi_dist, -math.log(1 - (old_mesh.ni[index, level] + self.dni[index, level])*Gamma_tot*cell_vol/(Gamma_pi[level]*photon.w))/Gamma_tot)

        if photoi_dist < photon.timestep_dist and photoi_dist > 0:
        #    rem_timestep_dist = photon.timestep_dist - photoi_dist
        #    photon.timestep_dist = photoi_dist
            defined = True
        else:
            defined = False

        # Move the particle, updating the cell tally in the process
        self.particle_tally[index] -= 1

        x_edges = self.get_edges_dim(1)
        z_edges = self.get_edges_dim(0)
        s = photon.move(x_edges[cell_i:(cell_i + 2)], z_edges[(cell_k):(cell_k + 2)], photoi_dist)

        #if defined:
        #    photon.timestep_dist += rem_timestep_dist

        outside_bounds = self.check_boundary_conditions(photon)
        if not outside_bounds:
            self.particle_tally[self.get_index(photon.cell_i, photon.cell_k)] += 1

        # Calculate the flux from implicit capture of this photon
        if Gamma_tot < 1e-8:
            dflux = s*photon.w/(cell_vol*dt)
        else:
            dflux = 1/(Gamma_tot*cell_vol*dt)*photon.w*(1 - math.exp(-Gamma_tot*s))

        if np.sum(Gamma_ib) < 1e-8:
            brem_flux = dflux
        else:
            brem_flux = 1/(np.sum(Gamma_ib)*cell_vol*dt)*photon.w*(1 - math.exp(-np.sum(Gamma_ib)*s))

        group = self.find_group(Constants.h*photon.nu)
        self.multigroup_flux[index, group] += dflux*Constants.h*photon.nu/Constants.c
        self.flux[index] += dflux
        self.energy_density[index] += dflux*Constants.h*photon.nu/Constants.c

        tot_photoi = 0
        for level in range(self.N_levels):
            # Calculate the number of photoionized particles for this ionization state
            # If this would photoionize more than remaining particles, reduce to zero instead
            photoi = min(old_mesh.ni[index, level] + self.dni[index, level], Gamma_pi[level]*dflux*dt)
            #photoi = Gamma_pi[level]*dflux*dt
            if photoi_dist <= 0 or (photoi > old_mesh.ni[index, level] + self.dni[index, level] and abs(old_mesh.ni[index, level] + self.dni[index, level] - photoi) > 1e-4*max(abs(old_mesh.ni[index, level] + self.dni[index, level]), abs(photoi))):
                print("Current atom density: ")
                print(old_mesh.ni[index, level])
                print("Remaining atom density: ")
                print(old_mesh.ni[index, level] + self.dni[index, level])
                print("Photoionization density: ")
                print(photoi)
                print("Photoionization cross section: ")
                print(Gamma_pi[level])
                print("Flux: ")
                print(dflux)
                print("Remaining photon distance: ")
                print(photon.timestep_dist)
                print("Maximum distance to travel without complete ionization: ")
                print(photoi_dist)
                print("Distance traveled: ")
                print(s)
                print("Distance calculated from photoionization: ")
                print(-math.log(1 - (photoi)*Gamma_tot*cell_vol/(Gamma_pi[level]*photon.w))/Gamma_tot)
                assert 0

            self.dni[index, level] -= photoi
            self.dni[index, level + 1] += photoi
            self.photoi_rate[index, level] += photoi
            self.dEpi[index] += photoi*(Constants.h*photon.nu - self.cell_mats[index].Eth[level])
            self.photoi_absorb_tot[index] += photoi*Constants.h*photon.nu
            self.dEib[index] += Gamma_ib[level]*dflux*dt*Constants.h*photon.nu
            #self.dEib[index] += Gamma_ib[level]*brem_flux*dt*Constants.h*photon.nu

            tot_photoi += photoi

        #photon.reduce_weight(s, np.sum(Gamma_ib))
        #photon.w -= tot_photoi*cell_vol
        photon.reduce_weight(s, Gamma_tot)

        return outside_bounds
    
    def get_index(self, cell_i, cell_k):
        return self.cells_per_dim[0]*cell_i + cell_k
    
    def get_cell_indices(self, cell):
        cell_i = int(cell/self.cells_per_dim[0])
        cell_k = cell % self.cells_per_dim[0]

        return cell_i, cell_k

    def get_edges_dim(self, dim):
        start = 0
        end = self.cells_per_dim[0] + 1
        for i in range(dim):
            start += self.cells_per_dim[i] + 1
            end += self.cells_per_dim[i + 1] + 1

        return self.cell_edges[start:end]
    
    def get_widths_dim(self, dim):
        start = 0
        end = self.cells_per_dim[0]

        for i in range(dim):
            start += self.cells_per_dim[i]
            end += self.cells_per_dim[i + 1]

        return self.cell_widths[start:end]
    
    def get_centers_dim(self, dim):
        start = 0
        end = self.cells_per_dim[0]

        for i in range(dim):
            start += self.cells_per_dim[i]
            end += self.cells_per_dim[i + 1]

        return self.cell_centers[start:end]

    def check_boundary_conditions(self, p):
        index_list = np.array([p.cell_k, p.cell_i])
        for i in range(self.dim):
            if index_list[i] < 0:
                match self.bcs[i*2]:
                    case 0:
                        return True
                    case 1:
                        p.mu = p.mu*(-1)**(i == 0)
                        p.phi = math.pi*(i == 1) + p.phi
                        p.phi -= 2*math.pi*(p.phi >= 2*math.pi)
                        p.cell_i += (i == 1)
                        p.cell_k += (i == 0)
            elif index_list[i] >= self.cells_per_dim[i]:
                match self.bcs[i*2 + 1]:
                    case 0:
                        return True
                    case 1:
                        p.mu = p.mu*(-1)**(i == 0)
                        p.phi = math.pi*(i == 1) + p.phi
                        p.phi -= 2*math.pi*(p.phi >= 2*math.pi)
                        p.cell_i -= (i == 1)
                        p.cell_k -= (i == 0)

        return False

    def calculate_eii(self, old_mesh, dt):
        # This function calculates the ionization from electron impact
        for cell in range(self.N_cells):
            for level in range(self.N_levels):
                eii_rate = min(self.ne[cell]*self.ni[cell, level]*self.cell_mats[cell].sigma_n(self.Te[cell], level)*dt, old_mesh.ni[cell, level] + self.dni[cell, level])

                self.dni[cell, level] -= eii_rate
                self.dni[cell, level + 1] += eii_rate
                self.dEeii[cell] += eii_rate*self.cell_mats[cell].Eth[level]

        return 0
    
    def calculate_rr_rate(self, old_mesh, dt):
        # This function calculates the rate of recombination in a photoionizing medium
        tot_recomb_rate = np.zeros((self.N_cells, ))
        for cell in range(self.N_cells):
            if self.atom_density[cell] > 1e12:
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
                    log_space_tbr = self.cell_mats[cell].tbr_n(self.Te[cell], level + 1) + 2*math.log(self.ne[cell]) + math.log(self.ni[cell, level + 1])
                    tbr_rate = min(math.exp(log_space_tbr)*dt, old_mesh.ni[cell, level + 1] + self.dni[cell, level + 1])
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
                    print(math.exp(log_space_tbr)*dt)
                    print("Remaining ion density: ")
                    print(old_mesh.ni[cell, level + 1] + self.dni[cell, level + 1])

                    assert 0

                self.dni[cell, level] += tbr_rate
                self.dni[cell, level + 1] -= tbr_rate
                self.dEtbr[cell] += tbr_rate*self.cell_mats[cell].Eth[level]

        return 0
    
    def source_particles_recombination(self, rng, census, cutoff, dt):
        # This function adds particles to a census after sourcing them from recombination
        cum_dErr = np.zeros((self.N_cells, ))
        for cell in range(self.N_cells):
            cell_i, cell_k = self.get_cell_indices(cell)
            cell_vol = self.cell_widths[cell_k]
            lin_index = 0
            for i in range(self.dim - 1):
                lin_index += self.cells_per_dim[i]
                cell_vol *= self.cell_widths[lin_index + cell_i]
            for level in range(self.N_levels):
                for particle in range(self.N_recomb[cell, level]):
                    x_pos_in_cell = self.get_edges_dim(1)[cell_i] + rng.random()*self.get_widths_dim(1)[cell_i]
                    z_pos_in_cell = self.get_edges_dim(0)[cell_k] + rng.random()*self.get_widths_dim(0)[cell_k]
                    mu = rng.random()*2 - 1
                    phi = rng.random()*2*math.pi
                    wgt = self.recomb_rate[cell, level]*cell_vol/(self.N_recomb[cell, level])
                    start_time = dt*rng.random()
                    xi = rng.random()

                    if wgt > cutoff:
                        cum_dErr[cell] += wgt*(self.cell_mats[cell].Eth[level] + xsf.sample_maxwellian(self.Te[cell], xi))

                        nu = (self.cell_mats[cell].Eth[level] + xsf.sample_maxwellian(self.Te[cell], xi))/Constants.h
                        photon = Particle(x_pos_in_cell, z_pos_in_cell, mu, phi, nu, wgt, cell_i, cell_k, (dt - start_time)/dt*Constants.c*dt)

                        census.append(photon)
                        self.particle_tally[cell] += 1

        return census

################### NEW - BREMSSTRAHLUNG SOURCE ##################################################

    def source_particles_bremsstrahlung(self, rng, census, cutoff, dt):
        # TODO: Make sure weights are correct (are in units of particle number)
        for cell in range(self.N_cells):
            if self.atom_density[cell] > 1e12:
                cell_i, cell_k = self.get_cell_indices(cell)
                cell_vol = self.cell_widths[cell_k]
                lin_index = 0
                for i in range(self.dim - 1):
                    lin_index += self.cells_per_dim[i]
                    cell_vol *= self.cell_widths[lin_index + cell_i]
                unadj_wgt = dt*4*math.pi*cell_vol*Constants.sigma_SB*self.Te[cell]**4/(Constants.keV2GJ)
                #sum_brems_xs = 0.0
                for particle in range(self.bremsstrahlung_N):
                    x_pos_in_cell = self.get_edges_dim(1)[cell_i] + rng.random()*self.get_widths_dim(1)[cell_i]
                    z_pos_in_cell = self.get_edges_dim(0)[cell_k] + rng.random()*self.get_widths_dim(0)[cell_k]
                    mu = rng.random()*2 - 1
                    phi = rng.random()*2*math.pi
                    start_time = dt*rng.random()
                    
                    xi = rng.random(5)
                    nu = xsf.sample_blackbody(self.Te[cell], xi)
                    #fail_safe = 0
#
                    #while Constants.h*nu < 0.005 and fail_safe < 100:
                    #    xi = rng.random(5)
                    #    nu = xsf.sample_blackbody(self.Te[cell], xi)
                    #    fail_safe += 1

                    sigma_brem_tot = self.cell_mats[cell].ibrem_xs(self, cell, nu, 2)*np.sum(self.ni[cell, 1:])

                    #wgt = self.cell_mats[cell].ibrem_xs(self, cell, nu, 2)*self.ni[cell, level]*xsf.blackbody(nu, self.Te[cell])[0]*dt*self.cell_widths[cell]*4*np.pi/(self.bremsstrahlung_N*Constants.h)
                    #wgt = self.cell_mats[cell].ibrem_xs(self, cell, nu, 2)*self.ni[cell, level]*unadj_wgt*self.cell_widths[cell]/(self.bremsstrahlung_N*self.N_levels*Constants.h*nu)
                    #conv_const = 1
                    wgt = sigma_brem_tot*unadj_wgt/(self.bremsstrahlung_N*Constants.h*nu)
                    #sum_brems_xs += sigma_brem_tot

                    if wgt > cutoff:
                        photon = Particle(x_pos_in_cell, z_pos_in_cell, mu, phi, nu, wgt, cell_i, cell_k, (dt - start_time)/dt*Constants.c*dt)

                        census.append(photon)
                        self.dEb[cell] += Constants.h*photon.nu*photon.w/cell_vol
                        self.particle_tally[cell] += 1

                        #if Constants.h*photon.nu*photon.w > 1e16:
                        #    print("***Excessive Bremsstrahlung Emission***")
                        #    print("Photon Energy: ")
                        #    print(Constants.h*photon.nu)
                        #    print("Photon Weight: ")
                        #    print(photon.w)
                        #    print("Energy Change: ")
                        #    print(Constants.h*photon.nu*photon.w)
                        #    print("Bremsstrahlung Cross Section: ")
                        #    print(sigma_brem_tot)
                        #    print("Ionization state: ")
                        #    print(self.ni[cell, :])
                        #    print("Unadjusted Weight: ")
                        #    print(unadj_wgt)
                        #    print("Temperature: ")
                        #    print(self.Te[cell])
                        #    print()

                
            #for index in range(self.bremsstrahlung_N):
            #    #if sum_brems_xs != 0 : census[-index - 1].w /= sum_brems_xs
            #    # TODO: Divide by cell volume to get an energy density
            #    self.dEb[cell] += Constants.h*census[-index - 1].nu*census[-index - 1].w/cell_vol

        return census #, sum_brems_xs

#################################################################################################

    def alpha(self, start, end):
        alpha = np.zeros((self.N_cells, self.N_levels))
        
        for level in range(self.N_levels):
            alpha[:, level] = self.recomb_rate[:, level]/self.photoi_rate[:, level]
        
        return alpha[start:end, :]
    
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
        self.photoi_absorb_tot = np.zeros((self.N_cells, ))
        self.dEpi = np.zeros((self.N_cells, ))
        self.dErr = np.zeros((self.N_cells, ))
        self.dEib = np.zeros((self.N_cells, ))
        self.dEb = np.zeros((self.N_cells, ))
        self.dEeii = np.zeros((self.N_cells, ))
        self.dEtbr = np.zeros((self.N_cells, ))

        return 0
    
    def copy(self):
        copied_mesh = Mesh(self.dim, self.cell_edges, self.N_levels, self.cells_per_dim, self.cell_mats[0], self.atom_density, self.recombination_N_per_cell, self.bremsstrahlung_N, self.bcs)
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

        self.dT = (self.dEpi + self.dEib + self.dEtbr - self.dEeii - self.dEb - self.dErr - 1.5*self.Te*dne)/xsf.cv(self.Te, self.atom_density + self.ne)
        T_update = mesh_previt.dT + lam*(self.dT - mesh_previt.dT)
        self.Te = old_mesh.Te + T_update 