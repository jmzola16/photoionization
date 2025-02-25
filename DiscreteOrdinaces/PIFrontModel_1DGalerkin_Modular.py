# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:16:01 2024

This program uses a weighted residual method to model the propogation of a 
photoionization front in one dimension. This version is created to be supplied
with different parts, so that basis functions, limiters, and other aspects of 
the code can be easily swapped out.

@author: jzola2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

M2 = 100                # Number of spatial cells in R

class PIFront:
    def __init__(self, N, M, P, S, L, R, c, dt, t_max):
        """
        Parameters
        ----------
        N : Integer
            N is the number of angles for the Sn angular discretization
        M : Integer
            M is the number of spatial cells along the z-axis
        P : Integer
            P is the number of basis functions for the flux
        S : Integer
            S is the number of basis functions for the ionized species
        L : Double
            L is the length of the tube [cm]
        R : Double
            R is the radius of the tube [cm]
        dt : Double
            dt is the time step of the method [ns]
        t_max : Double
            t_max is the end time of the simulation [ns]

        Returns
        -------
        None.
        """
        # Define parameters
        self.N = N                                  # Number of discrete ordinances
        self.M = M                                  # Number of spatial cells along tube length
        self.P = P                                  # Number of flux basis functions
        self.S = S                                  # Number of ionization basis functions
        
        # Duct geometry
        self.L = L                                  # Duct length [cm]
        self.R = R                                  # Duct radius [cm]
        self.C = 2*np.pi*R                          # Duct circumference [cm]
        self.dz = L/M                               # Cell width [cm]

        self.z = np.linspace(0, L, M)               # Cell edges (length) [cm]
        self.rho = np.linspace(0, R, M2)            # Cell edges (radius) [cm]
        self.var_phi = np.linspace(0, 2*np.pi, M2)  # Angular discretization [cm]

        # Time stepping
        self.dt = dt
        self.t_max = t_max

        # Physical parameters
        self.wall_reflect = c            # Wall reflection coefficient
        self.c = 30                      # Speed of light [cm/ns]
        self.SIGMA_SB = 5.670374e-8      # Stefan-Boltzmann Constant [W/(m^2*K^4)]
        self.n = 2e20                    # Particle density [particles/cm^3]
        self.sigma_s = self.n*5.1e-27    # Scattering cross section [cm^2]
        self.kappa = 0.87e-18            # Photoionization cross section [cm^2]
        self.Ts = 0.1                    # Source temperature (keV)

        self.mu, self.w = np.polynomial.legendre.leggauss(N)

        # Define coupling tensors
        self.A = np.zeros((P, P))
        self.B = np.zeros((P, P))
        self.A[0, 0] = 2/(np.pi*R)
        self.B[0, 0] = 2/(np.pi*R)
        if P > 1:
            self.A[0, 1] = ((3*np.pi - (16/np.pi))*(9*np.pi**2 - 64)**(-0.5))/R
            self.A[1, 0] = (-16*(9*np.pi**2 - 64)**(-0.5))/(np.pi*R)
            self.A[1, 1] = (128*(9*np.pi**2 - 64)**(-1))/(np.pi*R)
            
            self.B[0, 1] = self.A[0, 1]
            self.B[1, 0] = self.A[1, 0]
            self.B[1, 1] = -(24*np.pi - (128/np.pi))*(9*np.pi**2 - 64)**(-1)/R
        if P > 2:
            u = 3*np.pi*(9*np.pi**2 - 64)**(-0.5)/R
            v = 8*R/(3*np.pi)
    
            q = 8*R*(9*np.pi/5*(9*np.pi**2 - 64)**(-1) - 2/(3*np.pi))
            p = R**(-2)*(1 - 576/25*(9*np.pi**2 - 64)**(-1))**(-0.5)
    
            self.A[0, 2] = -q*p + (v**2 + q*v - 1/(u**2))*p*self.C/(np.pi**2*R**2)
            self.A[1, 2] = (2*p/u) - (v**2 + q*v - 1/(u**2))*u*v*p*self.C/(np.pi**2*R**2)
            self.A[2, 0] = (v**2 + q*v - 1/(u**2))*p*self.C/(np.pi**2*R**2)
            self.A[2, 1] = -(v**2 + q*v - 1/(u**2))*u*v*p*self.C/(np.pi**2*R**2)
            self.A[2, 2] = (v**2 + q*v - 1/(u**2))**2*p**2*self.C/(np.pi**2*R**2)
            
        self.iota = np.zeros((self.S, self.P))
        self.gamma = np.zeros((self.P, self.P, self.S))
        self.epsilon = np.zeros((self.S, self.P, self.S))

        # Create dependent variables
        self.psi = np.zeros((self.M, self.P, self.N)) #+ 1e19/2
        self.psi_old = np.zeros((self.M, self.P, self.N)) #+ 1e19/2

        self.Phi = np.zeros((M, P)) #+ 1e19
        self.Phi_guess = np.zeros((M, P))
        
        self.wall_scatter = np.zeros((M, P))
        self.wall_scatter_guess = np.zeros((M, P))

        self.nb = np.zeros((self.M, self.S))
        self.nb_old = np.zeros((self.M, self.S))
        self.nb_guess = np.zeros((self.M, self.S))
        self.nb_guess[0, 0] = 1

    ##### BASIS FUNCTIONS #####
    def compute_basis_integrals(self, alpha, *args, **kwargs):
        self.alpha = alpha
        for key, value in kwargs.items():
            # Determine method for integration
            if key == 'quadriture':
                if value == 'trapezoid':
                    # Use trapezoid rule for integration
                    def integrator(r_range, phi_range, integrand):
                        # Declare a wrapper function which calls scipy.integrate's trapezoid 
                        # function for a particular integrand over the domain r, phi
                        int_of_phi = np.zeros((len(phi_range), ))
                        for p in range(len(phi_range)):
                            int_of_phi[p] = integrate.trapezoid(integrand(r_range, phi_range[p]), r_range)
                            
                        return integrate.trapezoid(int_of_phi, phi_range)
                else:
                    # Use gauss quadriture for integration
                    def integrator(r_range, phi_range, integrand):
                        # Declare a wrapper function which calls scipy.integrate's dblquad
                        # function for a partticular integrand. 
                        return integrate.dblquad(integrand, 0, 2*np.pi, 0, self.R)[0]
        if len(args) > 0:
            # A second basis is given
            self.beta = args[0]
            temp_S = self.S
            
            # Calculate two of the required tensors
            for p in range(self.S):
                for q in range(self.P):
                    for s in range(self.S):
                        self.epsilon[p, q, s] = 1/(2*np.pi)*integrator(self.rho, self.var_phi, lambda r, phi : r*self.alpha(r, phi, q + 1)*self.beta(r, phi, p + 1)*self.beta(r, phi, s + 1))
                        if s == 0:
                            self.iota[p, q] = 1/(2*np.pi)*integrator(self.rho, self.var_phi, integrand=lambda r, phi : r*self.alpha(r, phi, q + 1)*self.beta(r, phi, p + 1))
        else:
            # Only one basis is given
            self.beta = alpha
            self.iota = np.eye(self.P)
        
            # The number of basis functions are the same
            temp_S = self.P
        
            # These two tensors are the same
            self.gamma = np.zeros(self.P, self.P, self.P)
            self.epsilon = self.gamma
    
        # Calculate the final tensor. This tensor can be calculated only once for 
        # either case.
        for p in range(self.P):
            for q in range(self.P):
                for s in range(temp_S):
                    self.gamma[p, q, s] = 1/(np.pi*self.R**2)*integrator(self.rho, self.var_phi, integrand=lambda r, phi : r*self.alpha(r, phi, p + 1)*self.alpha(r, phi, q + 1)*self.beta(r, phi, s + 1))

    ##### BOUNDARY CONDITION ######

    def set_boundary_condition(self, alpha, Ts, shape, pltit=False):
        # Calculate a scalar flux from a given source temperature
        a = 0.01372
        self.Ts = Ts
        self.f = np.zeros((self.P, self.N))
        peak = (a*self.c*Ts**3/(4*2.71*1.602e-25)) # Equation (5) Drake 2016
        # if shape=='constant':
        #     # The boundary condition is a constant flux across all angles
        #     self.f[0, :] = np.ones((self.N, ))*peak
        # if shape=='gaussian':
        #     # The boundary condition is a gaussian distribution of flux in all angles
        #     std_dev = self.R/3
        for i in range(self.P):
            # Use a least squares approximation to approximate the source function, given by shape
            const = (1/(np.pi*self.R**2))*integrate.dblquad(lambda r, phi : r*shape(r)*alpha(r, phi, i + 1), 0, 2*np.pi, 0, self.R)[0]
            self.f[i, :] = const*np.ones((self.N, ))*peak
        
        if pltit:
            num = 100
            approx = np.zeros((num, ))
            rho = np.linspace(0, self.R, num)
            plt.plot(rho, peak*shape(rho), label='True Source')
            for i in range(self.P):
                for j in range(num):
                    approx[j] += np.sum(self.f[i, :]*self.w)*(1/(4*np.pi))*integrate.quad(lambda phi : alpha(rho[j], phi, i + 1), 0, 2*np.pi)[0]
            plt.plot(rho, approx, label='Approximate Source')
            plt.xlabel('Radial distance [cm]')
            plt.ylabel('Incoming Flux [# of particles/(cm^2 s)]')
            plt.legend()
            plt.show()

    ##### TRANSPORT SWEEP/ANGULAR SOLVE #####

    def transport_sweep(self):
        for j in range(self.N):
            # Define matrix to solve for coupled terms in each angle
            Q = np.zeros((self.P, self.P))
            # Loop over each WR node
            for k in range(self.P):
                for l in range(self.P):
                    if k != l:
                        # Node being solved for (on diagonal)
                        Q[k, l] = -self.kappa*np.sum(self.gamma[k, l, :]*self.nb_guess[0, :]) + self.A[k, l]*(1 - self.mu[j]**2)**0.5
                    else:
                        # Contributions from other nodes (off diagonal)
                        Q[k, l] = 1/(self.c*self.dt) + abs(self.mu[j])/self.dz + self.kappa*self.n - self.kappa*np.sum(self.gamma[k, l, :]*self.nb_guess[0, :]) + (1 - self.mu[j]**2)**(0.5)*self.A[k, l] + self.sigma_s
            
            # Knowns for each equation
            rhs = self.sigma_s/2*self.Phi_guess[0, :] + 1/(self.c*self.dt)*self.psi_old[0, :, j] + self.mu[j]/self.dz*self.f[:, j]*(self.mu[j] > 0) + 2*self.wall_reflect/np.pi*(1 - self.mu[j]**2)**0.5*np.matmul(self.B, self.wall_scatter_guess[0, :])
            
            self.psi[0, :, j] = np.linalg.solve(Q, rhs)
                        
        # Loop over all spatial cells
        for i in range(1, self.M):
            # Loop over all angles
            for j in range(self.N):
                # Set up system of equations
                Q = np.zeros((self.P, self.P))
                # Loop over WR nodes
                for k in range(self.P):
                    for l in range(self.P):
                        if k != l:
                            # Contributions from other nodes (off diagonal)
                            Q[k, l] = -self.kappa*np.sum(self.gamma[k, l, :]*self.nb_guess[i, :]) + self.A[k, l]*(1 - self.mu[j]**2)**0.5
                        else:
                            # Node being solved for (on diagonal)
                            Q[k, l] = 1/(self.c*self.dt) + abs(self.mu[j])/self.dz + self.kappa*self.n - self.kappa*np.sum(self.gamma[k, l, :]*self.nb_guess[i, :]) + (1 - self.mu[j]**2)**(0.5)*self.A[k, l] + self.sigma_s
                
                # Knowns for each equation (depends on mu, because flux could be left going or right going)
                if (self.mu[j] > 0):
                    rhs = self.sigma_s/2*self.Phi_guess[i, :] + 1/(self.c*self.dt)*self.psi_old[i, :, j] + self.mu[j]/self.dz*self.psi[i - 1, :, j] + 2*self.wall_reflect/np.pi*(1 - self.mu[j]**2)**0.5*np.matmul(self.B, self.wall_scatter_guess[i, :])
                    self.psi[i, :, j] = np.linalg.solve(Q, rhs)
                else:
                    rhs = self.sigma_s/2*self.Phi_guess[self.M - i - 1, :] + 1/(self.c*self.dt)*self.psi_old[self.M - i - 1, :, j] - self.mu[j]/self.dz*self.psi[self.M - i, :, j] + 2*self.wall_reflect/np.pi*(1 - self.mu[j]**2)**0.5*np.matmul(self.B, self.wall_scatter_guess[self.M - i - 1, :])
                    self.psi[self.M - i - 1, :, j] = np.linalg.solve(Q, rhs)
                  
        self.wall_scatter = np.zeros((self.M, self.P))
        for k in range(self.P):
            # Compute scalar flux and wall scatter coefficient at each node
            self.Phi[:, k] = np.matmul(self.psi[:, k, :], self.w)
            for j in range(self.N):
                self.wall_scatter[:, k] += (1 - self.mu[j]**2)**(0.5)*self.psi[:, k, j]*self.w[j]
        
        # Loop over all spatial cells
        for i in range(self.M):
            # Loop over WR nodes
            T = np.zeros((self.S, self.S))
            for k in range(self.S):
                for l in range(self.S):
                    if k != l:
                        # Contributions from other nodes (off diagonal)
                        T[k, l] = self.kappa*(np.sum(self.epsilon[k, :, l]*self.Phi[i, :]))
                    else:
                        # Node being solved for (on diagonal)
                        T[k, l] = 1/self.dt + self.kappa*(np.sum(self.epsilon[k, :, l]*self.Phi[i, :]))
            
            # Known quantities
            rhs = self.nb_old[i, :]/self.dt + self.kappa*self.n*np.matmul(self.iota, self.Phi[i, :])
            
            self.nb[i, :] = np.linalg.solve(T, rhs)

    def nonlinear_solve(self, tol, max_it):
        ##### NONLINEAR SOLVE #####
        it = 0
        err = self.n*(np.ones((self.P + self.S, )))
        while (max(err) > self.n*tol and it < max_it):
            self.transport_sweep()
            
            # Update error by comparing new solutions to guesses
            for s in range(self.S):
                err[s] = np.linalg.norm(self.nb[:, s] - self.nb_guess[:, s])
            for p in range(self.P):
                err[self.S + p] = np.linalg.norm(self.Phi[:, p] - self.Phi_guess[:, p])
            
            # Guess new solution to check convergence
            self.nb_guess[:] = self.nb[:]
            self.Phi_guess[:] = self.Phi[:]
            self.wall_scatter_guess[:] = self.wall_scatter[:]
                
            it += 1
            
        self.nb_old[:] = self.nb[:]
        self.psi_old[:] = self.psi[:]

    def time_step(self, tol, max_it, plt_int):
        ##### TIME STEPPING #####
        plt_step = plt_int
        
        self.front_location = np.zeros((int(self.t_max/self.dt) + 1, ))
        for t in range(int(self.t_max/self.dt) + 1):
            self.nonlinear_solve(tol, max_it)
            
            ions = np.zeros((self.M, ))
            for i in range(self.S):
                ions += self.nb[:, i]*self.beta(0, self.var_phi, i + 1)
                
            ions /= self.n
                
            index = 0
            
            while ions[index] > 0.1 and index < self.M - 1:
                index += 1
                
            self.front_location[t] = self.z[index]
            
            if t == plt_step or t == (int(self.t_max/self.dt)):
                self.plot_state(plt_step)
                self.plot_state_contour(plt_step)
                plt_step += plt_int
                
        self.plot_front_location()

    def plot_state(self, plt_step):
        ##### PLOTTING #####
        
        flux = np.zeros((M2, self.M))
        ions = np.zeros((M2, self.M))
        for edge in range(M2):
            for i in range(self.P):
                alpha_phi = 1/(2*np.pi)*integrate.quad(lambda phi : self.alpha(self.rho[edge], phi, i + 1), 0, 2*np.pi)[0]
                flux[edge, :] += self.Phi[:, i]*alpha_phi
                
            for i in range(self.S):
                ions[edge, :] += self.nb[:, i]*self.beta(self.rho[edge], self.var_phi, i + 1)
        ions = ions/self.n
        
        fig, axs = plt.subplots(3, 1)
        im = axs[0].pcolormesh(self.z, self.rho, flux)
        if (self.R/self.L) > 0.05 and (self.R/self.L) < 20:
            axs[0].axis('equal')
        axs[0].set_ylabel('Radius [cm]')
        axs[0].set_title('Scalar flux at t=' + str(plt_step*self.dt))
        axs[0].set_xticks([])
        fig.colorbar(im, ax=axs[0])
        
        im = axs[1].pcolormesh(self.z, self.rho, (((4*flux*2.7*self.Ts*1.6022e-25)/(0.01372*self.c))**(0.25))/self.Ts)
        if (self.R/self.L) > 0.1 and (self.R/self.L) < 10:
            axs[1].axis('equal')
        axs[1].set_ylabel('Radius [cm]')
        axs[1].set_title('T/Ts')
        axs[1].set_xticks([])
        fig.colorbar(im, ax=axs[1])
        
        im = axs[2].pcolormesh(self.z, self.rho, ions)
        if (self.R/self.L) > 0.1 and (self.R/self.L) < 10:
            axs[2].axis('equal')
        plt.xlabel('Distance [cm]')
        axs[2].set_ylabel('Radius [cm]')
        plt.title('Ionization fraction at t=' + str(plt_step*self.dt))
        fig.colorbar(im, ax = axs[2])
        plt.show()
        
    def plot_state_contour(self, plt_step):
        ions = np.zeros((M2, self.M))
        for edge in range(M2):                
            for i in range(self.S):
                ions[edge, :] += self.nb[:, i]*self.beta(self.rho[edge], self.var_phi, i + 1)
        ions = ions/self.n
        
        plt.contour(self.z, self.rho, ions, np.array([0.1, 0.5, 1]))
        plt.title('Contours of ionization fraction at t=' + str(plt_step*self.dt))
        plt.ylabel('Radius [cm]')
        plt.xlabel('Distance [cm]')
        plt.show()    
    
    def plot_front_location(self):
        plt.plot(np.linspace(self.dt, self.t_max, int(self.t_max/self.dt) + 1), self.front_location)
        plt.title('Distance over time of Photoionization Front')
        plt.xlabel('Time [ns]')
        plt.ylabel('Front location [cm]')
        plt.show()