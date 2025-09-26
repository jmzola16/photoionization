# This file defines a particle class, which 
from numba.experimental import jitclass
from numba import float64, int64
import numpy as np
import Constants

spec = [
    ('x', float64),
    ('mu', float64),
    ('nu', float64),
    ('w', float64),
    ('cell', int64),
    ('timestep_dist', float64),
]

#@jitclass(spec)
class Particle:
    def __init__(self, x, mu, nu, w, cell, timestep_dist):
        self.x = x
        self.mu = mu
        self.w = w
        self.nu = nu
        self.cell = cell
        self.timestep_dist = timestep_dist
        
    def move(self, bounds):
        # Move a particle along its direction vector
        path_length = (bounds[1] - self.x)/self.mu*(self.mu > 0) + (self.x - bounds[0])/self.mu*(self.mu < 0)
        if abs(path_length) < self.timestep_dist:
            # If the particle can reach the border of the cell within the current time step, increment its cell
            self.x += path_length*(abs(self.mu))
            self.timestep_dist -= abs(path_length)
            self.cell += int(np.sign(self.mu))
        else:
            # Else move the particle within the cell
            path_length = self.timestep_dist
            self.x += self.timestep_dist*self.mu
            self.timestep_dist = 0.0
            
        # Return the distance travelled
        return abs(path_length)
            
    def reduce_weight(self, path_length, Sigma):
        # Reduce the particle length by the path length traveled
        self.w *= np.exp(-Sigma*path_length)

    def copy(self):
        # Return a deep copy of a particle object, for use in copying the census list
        copied_particle = Particle(self.x, self.mu, self.nu, self.w, self.cell, self.timestep_dist)
        return copied_particle
    
    def roulette(self, roulette_pool, particle_tally):
        self.w += roulette_pool[self.cell]/(particle_tally[self.cell]*Constants.h*self.nu)