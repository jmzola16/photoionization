# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 08:33:20 2024

This program solves for the scalar flux and ionization fraction in a photoionization
front using an Sn method with source iteration

@author: James Zola
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import CrossSectionFunctions as xsf

N = 8
M = 100
R = 0.1 # cm
L = 0.7 # cm
c = 30 # cm/ns
dt = 0.001 # ns
t_max = 2
dz = L/M
z = np.linspace(0, L, M)
n = 2e20 #1
T = 0.100 # keV
Te_0 = 1e-5 # Initial electron temperature [keV]
Gamma = np.array([xsf.pi_n1(2.71*T), xsf.pi_n2(2.71*T), xsf.pi_n3(2.71*T), xsf.pi_n4(2.71*T)]) # cm^2
levels = len(Gamma) + 1 # Number of photoionization levels
rr = np.zeros((levels - 1, ))
Eth = np.array([14.53, 29.60, 47.45, 77.47])*1e-3
keV2J = 1.602e-19
k_B = 8.617e-8
Na = 6.022e23
rho_N = 0.02802
cv = 741/(keV2J*k_B)*(n/(Na)*rho_N)  # Material specific heat [keV/(keV*cm^3)]
sigma_s = n*5.1e-27 # cm^2
f = (np.ones((N, ))*0.01372*c*T**3)/(4*2.71*1.602e-25)
a = 2/(R*(np.pi))

mu, w = np.polynomial.legendre.leggauss(N)

ni_old = np.zeros((M, levels))
ni_old[:, 0] = np.ones((M, ))*n
ni_guess = np.zeros((M, levels))
ni_guess[:, 0] = np.ones((M, ))*n
ni_guess[1, 1] = 1
ni = np.zeros((M, levels))
ni[:, 0] = np.ones((M, ))*n
temp = np.zeros((M, ))
ne = np.zeros((M, ))
ne_old = np.zeros((M, ))

psi_old = np.zeros((M, N))
psi = np.zeros((M, N))
phi = np.matmul(psi, w)
phi_old = np.matmul(psi, w)
phi_it = np.matmul(psi, w)

Te_old = np.ones((M, ))*Te_0
Te = np.ones((M, ))*Te_0
Te_it = np.ones((M, ))*Te_0

plt.subplots(2, 1)

err_ni = np.zeros((levels, ))

for t in range(int(t_max/dt) + 1):
    err1 = n
    err2 = n
    tol = n*1e-8
    it = 0
    while((err1 > tol or err2 > tol) and it < 50):
        psi[0, int(N/2):] = (1/(c*dt)*psi_old[0, int(N/2):] + mu[int(N/2):]*f[int(N/2):]/(dz) + sigma_s/2*phi_it[0])/(1/(c*dt) + mu[int(N/2):]/(dz) + (1 - mu[int(N/2):]**2)**(0.5)*a + sum(ni_guess[0, :-1]*Gamma) + sigma_s) # (n - nb_old[0])*kappa/(1 + kappa*dt*phi_old[0])
        psi[-1, :int(N/2)] = (1/(c*dt)*psi_old[-1, :int(N/2)] + sigma_s/2*phi_it[-1])/(1/(c*dt) - mu[:int(N/2)]/(dz) + (1 - mu[:int(N/2)]**2)**(0.5)*a + sum(ni_guess[-1, :-1]*Gamma) + sigma_s) # (n - nb_old[-1])*kappa/(1 + kappa*dt*phi_old[-1])
        for j in range(1, M):
            psi[j, int(N/2):] = (1/(c*dt)*psi_old[j, int(N/2):] + mu[int(N/2):]*psi[j - 1, int(N/2):]/(dz) + sigma_s/2*phi_it[j])/(1/(c*dt) + mu[int(N/2):]/(dz) + (1 - mu[int(N/2):]**2)**(0.5)*a + sum(ni_guess[j, :-1]*Gamma) + sigma_s) # (n - nb_old[j])*kappa/(1 + kappa*dt*phi_old[j])
            psi[M - j - 1, :int(N/2)] = (1/(c*dt)*psi_old[M - j, :int(N/2)] - mu[:int(N/2)]*psi[M - j, :int(N/2)]/(dz) + sigma_s/2*phi_it[M - j])/(1/(c*dt) - mu[:int(N/2)]/(dz) + (1 - mu[:int(N/2)]**2)**(0.5)*a + sum(ni_guess[M - j, :-1]*Gamma) + sigma_s) # (n - nb_old[j])*kappa/(1 + kappa*dt*phi_old[M - j])
            
        phi_it = np.matmul(psi, w)
        
        for j in range(M):
            # if ne[j] > 1e-10:
            #     temp_T = Te_it[j]/ne[j]
            # else:
            #     temp_T = 1e-5
            # rr = np.array([xsf.rr_n1(Te_it), xsf.rr_n2(Te_it), xsf.rr_n3(Te_it), xsf.rr_n4(Te_it)])
            
            # Define a matrix which relates the rate of change of the ionization levels
            if levels > 2:
                rate_matrix = np.zeros((levels - 1, levels - 1))
                rate_matrix[0, 0] = 1/dt + Gamma[1]*phi_it[j] + rr[0]*ne[j]
                rate_matrix[0, 1] = -rr[1]*ne[j]
                rate_matrix[0, :] += Gamma[0]*phi_it[j]
            
                RHS = np.zeros((levels - 1, ))
                RHS[0] = Gamma[0]*n*phi_it[j] + ni_old[j, 1]/dt
            
                for i in range(1, levels - 2):
                    rate_matrix[i, i - 1] = -Gamma[i]*phi_it[j]
                    rate_matrix[i, i] = 1/dt + Gamma[i + 1]*phi_it[j] + rr[i]*ne[j]
                    rate_matrix[i, i + 1] = -rr[i + 1]*ne[j]
                
                    RHS[i] = (ni_old[j, i])/dt
                
                rate_matrix[-1, -1] = 1/dt + rr[-1]*ne[j]
                rate_matrix[-1, -2] = -Gamma[-1]*phi_it[j]
            
                RHS[-1] = (ni_old[j, -1])/dt
        
                ni[j, 1:] = np.linalg.solve(rate_matrix, RHS)
                ni[j, 0] = n - np.sum(ni[j, 1:])
            else:
                ni[j, 1] = ((1/dt)*ni_old[j, 1] + n*Gamma[0]*phi_it[j])/(1/dt + Gamma[0]*phi_it[j] - rr[0]*ne[j])
                ni[j, 0] = n - ni[j, 1]
        
        ne = np.matmul(ni, np.arange(levels))
        
        Te_it[:] = Te_old + (phi_it*dt*np.matmul(ni[:, :-1], (2.71*T - Eth)*Gamma) - ne*dt*np.matmul(ni[:, 1:], (2.71*T - Eth)*rr))/(cv*dz)
        
        err1 = np.linalg.norm(phi - phi_it)
        
        for i in range(levels):
            err_ni[i] = np.linalg.norm(ni[:, i] - ni_guess[:, i])
            
        err2 = max(err_ni) #np.linalg.norm(nb - nb_guess)
        
        ni_guess[:] = ni[:]
        
        phi[:] = phi_it[:]
        
        Te[:] = Te_it[:]
        
        it += 1
    
     
    #temp[:] = nb[:]
    #nb = nb_old + dt*kappa*(n - nb_old)*phi/(1 + kappa*dt*phi_old)
    #nb_old[:] = temp[:]

    psi_old[:] = psi[:]
    psi = np.zeros((M, N))
    phi_old[:] = phi[:]
    
    ni_old[:] = ni[:]
    
    ne_old[:] = ne[:]
    
    Te_old[:] = Te[:]
    
    # if (dt*t > p/c and p < 7):
    #     plt.figure(1)
    #     plt.plot(z, nb/n, label='t=' + str(dt*t))
    #     plt.figure(2)
    #     plt.plot(z, phi, label='t=' + str(dt*t))

    #     p += 1
    
    # if t == 100:
    #     plt.subplot(2, 1, 2)
    #     #plt.plot(z, Te, label='t=0.1')
    #     plt.plot(z, ni[:, 1]/n, label='t=0.1')
    #     plt.subplot(2, 1, 1)
    #     plt.plot(z, phi*2.71*T/c, label='t=0.1')
    # elif t == 450:
    #     plt.subplot(2, 1, 2)
    #     #plt.plot(z, Te, label='t=0.45')
    #     plt.plot(z, ni[:, 1]/n, label='t=0.45')
    #     plt.subplot(2, 1, 1)
    #     plt.plot(z, phi*2.71*T/c, label='t=0.45')
    # elif t == 750:
    #     plt.subplot(2, 1, 2)
    #     #plt.plot(z, Te, label='t=0.75')
    #     plt.plot(z, ni[:, 1]/n, label='t=0.75')
    #     plt.subplot(2, 1, 1)
    #     plt.plot(z, phi*2.71*T/c, label='t=0.75')
    # elif t == 1000:
    #     plt.subplot(2, 1, 2)
    #     #plt.plot(z, Te, label='t=1')
    #     plt.plot(z, ni[:, 1]/n, label='t=1')
    #     plt.subplot(2, 1, 1)
    #     plt.plot(z, phi*2.71*T/c, label='t=1')
    # elif t == 2000:
    #     plt.subplot(2, 1, 2)
    #     #plt.plot(z, Te, label='t=2')
    #     plt.plot(z, ni[:, 1]/n, label='t=2')
    #     plt.subplot(2, 1, 1)
    #     plt.plot(z, phi*2.71*T/c, label='t=2')


#plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(z, phi*2.71*T/c, label='t='+str(round(t_max, 2)))
plt.title('Energy Density and Ion Fraction at t = ' + str(round(t_max, 2)) + ' ns')
#plt.title('Energy Density and Temperature over Time')
plt.ylabel('Energy Density, [keV/$cm^3$]')
#plt.legend()

plt.subplot(2, 1, 2)
# plt.plot(z, ni[:, 1]/n, label='t=' + str(round(t_max, 2)))
# plt.ylabel('Ion Fraction [-]')
# #plt.plot(z, Te, label='t='+str(round(t_max, 2)))
# #plt.ylabel('Electron temperature [eV]')
# plt.xlabel('z-location [cm]')
# plt.show()
        
# plt.figure(1)
#fig = plt.figure(2)
plt.plot(z, ni/n)   
plt.legend(['n' + str(i + 1) for i in range(levels)])
# plt.title('Ionization fraction at t=' + str(round(t_max, 2)) + ' ns')
plt.xlabel('z-location [cm]')
plt.ylabel('Ion fraction [-]')
plt.show()