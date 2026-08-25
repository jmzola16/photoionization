import numpy as np
import sys
sys.path.append("..")
from abc import ABC, abstractmethod
import Constants
import CrossSectionFunctions as xsf
import math
from Particle import Particle
from numba.typed import List
from numba import float64, int32
from numba.experimental import jitclass

class Source(ABC):
    @abstractmethod
    def source_particles(self, rng, census, mesh, dt, N_source, Ts, t):
        pass

spec_point = [
    ('x_loc', float64),
    ('z_loc', float64),
    ('mu_range', float64[:]),
    ('phi_range', float64[:])
]

class Point_Source(Source):
    def __init__(self, x_loc, z_loc, mu_range, phi_range):
        self.x_loc = x_loc
        self.z_loc = z_loc
        self.mu_range = mu_range
        self.phi_range = phi_range

    def source_particles(self, rng, census, mesh, dt, N_source, Ts, t):
        x_edges = mesh.get_edges_dim(1)
        z_edges = mesh.get_edges_dim(0)
        solid_angle = (self.mu_range[1] - self.mu_range[0])*(self.phi_range[1] - self.phi_range[0])

        emission_rate = dt*solid_angle*Constants.sigma_SB*Ts(t)**4/(Constants.keV2GJ)
        w0 = emission_rate/N_source
        cutoff = 1e-10*w0
        source_energy = 0
        for i in range(N_source):
            # For each particle to source, sample parameters
            start_time = dt*rng.random()
            xi = rng.random(5)
            x = self.x_loc
            z = self.z_loc
            cell_i = np.searchsorted(x_edges, x) - 1
            cell_k = np.searchsorted(z_edges, z) - 1

            mu = rng.random()*(self.mu_range[1] - self.mu_range[0]) + self.mu_range[0]
            phi = rng.random()*(self.phi_range[1] - self.phi_range[0]) + self.phi_range[0]
            nu = xsf.sample_blackbody(Ts(t), xi)

            timestep_dist = Constants.c*dt

            # Construct the photon and add to census, counting particles in each cell
            photon = Particle(x, z, mu, phi, nu, w0/(Constants.h*nu), cell_i, cell_k, (dt - start_time)/dt*timestep_dist)
            census.append(photon)

            source_index = mesh.get_index(cell_i, cell_k)
            mesh.particle_tally[source_index] += 1

            source_energy += photon.w*Constants.h*photon.nu

        return census, mesh, cutoff, source_energy

spec_plane = [
    ('plane_loc', float64),
    ('plane_range', float64[:]),
    ('plane_code', int32),
    ('source_face', int32)
] 

class Plane_Source(Source):
    def __init__(self, plane_loc, plane_range, plane_code, source_face):
        self.plane_loc = plane_loc          # The coordinate of the plane along the axis perpendicular to it's face
        self.plane_range = plane_range      # How far the plane extends
        self.plane_code = plane_code        # Which plane this plane is parallel to (0 for xy, 1 for yz, 2 for xz)
        self.source_face = source_face      # Whether the particles are source from the positive or negative face

    def source_particles(self, rng, census, mesh, dt, N_source, Ts, t):
        x_edges = mesh.get_edges_dim(1)
        z_edges = mesh.get_edges_dim(0)
        plane_area = self.plane_range[1] - self.plane_range[0]
        solid_angle = 2*math.pi

        emission_rate = dt*plane_area*solid_angle*Constants.sigma_SB*Ts(t)**4/(Constants.keV2GJ)
        w0 = emission_rate/N_source
        cutoff = w0*1e-10
        source_energy = 0

        for i in range(N_source):
            if self.plane_code == 0:
                # Source is parallel to xy plane
                x_loc = rng.random()*(self.plane_range[1] - self.plane_range[0]) + self.plane_range[0]
                z_loc = self.plane_loc
                mu = math.sqrt(rng.random())*self.source_face
                phi = rng.random()*2*math.pi
            elif self.plane_code == 1:
                # Source is parallel to yz plane
                x_loc = self.plane_loc
                z_loc = rng.random()*(self.plane_range[1] - self.plane_range[0]) + self.plane_range[0]

                # Sample the cosine with the x-axis according to the planar distribution
                cosphi = self.source_face*math.sqrt(rng.random())     # Omega x
                #phi_star = np.arccos(cosphi)
                #shift_phase = rng.random() < 0.5
                theta = rng.random()*2*math.pi
                mu = math.cos(theta)*math.sqrt(1 - cosphi**2)

                # Shift the phase with probability 0.5 to ensure entire plane face is covered
                phi = math.acos(cosphi/(math.sqrt(1 - mu**2)))
                #phi_star*(shift_phase) + (2*np.pi - phi_star)*(not shift_phase)
            else:
                print("Sources parallel to xz plane not yet implemented. Aborting...")
                assert 0

            start_time = rng.random()*dt
            xi = rng.random(5)
            if x_loc == x_edges[0]:
                cell_i = 0
            elif x_loc == x_edges[-1]:
                cell_i = len(x_edges) - 2
            else:
                cell_i = np.searchsorted(x_edges, x_loc) - 1
            if z_loc == z_edges[0]:
                cell_k = 0
            elif z_loc == z_edges[-1]:
                cell_k = len(z_edges) - 2
            else:
                cell_k = np.searchsorted(z_edges, z_loc) - 1

            nu = xsf.sample_blackbody(Ts(t), xi)

            timestep_dist = Constants.c*dt

            # Construct the photon and add to census, counting particles in each cell
            photon = Particle(x_loc, z_loc, mu, phi, nu, w0/(Constants.h*nu), cell_i, cell_k, (dt - start_time)/dt*timestep_dist)
            census.append(photon)

            source_index = mesh.get_index(cell_i, cell_k)
            mesh.particle_tally[source_index] += 1

            source_energy += photon.w*Constants.h*photon.nu

        return census, mesh, cutoff, source_energy