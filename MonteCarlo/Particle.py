# This file defines a particle class, which 
from numba.experimental import jitclass
from numba import float64, int64
import numpy as np
import Constants

spec = [
    ('x', float64),
    ('z', float64),
    ('mu', float64),
    ('phi', float64),
    ('nu', float64),
    ('w', float64),
    ('cell_i', int64),
    ('cell_k', int64),
    ('timestep_dist', float64)
]

@jitclass(spec)
class Particle:
    def __init__(self, x, z, mu, phi, nu, w, cell_i, cell_k, timestep_dist):
        self.x = x
        self.z = z
        self.mu = mu
        self.phi = phi
        self.nu = nu
        self.w = w
        self.cell_i = cell_i
        self.cell_k = cell_k
        self.timestep_dist = timestep_dist
        
    def move(self, x_bounds, z_bounds):
        # Move a particle along its direction vector
        cosphi = np.cos(self.phi)

        z_dist = (z_bounds[1] - self.z)/self.mu*(self.mu > 0) + (self.z - z_bounds[0])/self.mu*(self.mu < 0)
        x_dist = (x_bounds[1] - self.x)/(np.sqrt(1 - self.mu**2)*cosphi)*(cosphi > 0) + (self.x - x_bounds[0])/(np.sqrt(1 - self.mu**2)*cosphi)*(cosphi < 0)
        path_length = min(abs(z_dist), abs(x_dist))
        if path_length < self.timestep_dist:
            # If the particle can reach the border of the cell within the current time step, increment its cell
            if abs(x_dist) < abs(z_dist):
                # The particle reaches an x-boundary first
                self.cell_i += int(np.sign(cosphi))
            else:
                self.cell_k += int(np.sign(self.mu))
            
            self.x += path_length*np.sqrt(1 - self.mu**2)*cosphi
            self.z += path_length*self.mu
            self.timestep_dist -= path_length
        else:
            # Else move the particle within the cell
            path_length = self.timestep_dist
            self.x += path_length*np.sqrt(1 - self.mu**2)*cosphi
            self.z += self.timestep_dist*self.mu
            self.timestep_dist = 0.0
            
        # Return the distance travelled
        return path_length
            
    def reduce_weight(self, path_length, Sigma):
        # Reduce the particle length by the path length traveled
        self.w *= np.exp(-Sigma*path_length)

    def copy(self):
        # Return a deep copy of a particle object, for use in copying the census list
        copied_particle = Particle(self.x, self.z, self.mu, self.phi, self.nu, self.w, self.cell_i, self.cell_k, self.timestep_dist)
        return copied_particle
    
    def roulette(self, roulette_pool, particle_tally, cell_index):
        self.w += roulette_pool[cell_index]/(particle_tally[cell_index]*Constants.h*self.nu)