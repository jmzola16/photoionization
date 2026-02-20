import numpy as np
import sys
sys.path.append("..")
import CrossSectionFunctions as xsf
import Constants
import matplotlib.pyplot as plt
from matplotlib import colors
from Mesh import Mesh
from scipy import integrate, optimize
from Particle import Particle
from numba import typeof
from numba.typed import List

# Test sample blackbody and blackbody
#N = 1000000
#a = 0.01372
#c = 30.0
#keV2GJ = 1.602e-25
#T = 0.1
#h = 4.135e-9
#bins = np.linspace(0, 5e9, 10000)
#bin_centers = bins[:-1] + np.diff(bins)
#bin_widths = np.diff(bins)
#heights = np.zeros((len(bin_centers), ))
#rng = np.random.default_rng()
#analytic = xsf.blackbody(bin_centers, T)[0]/(a*c*T**4/(4*np.pi*keV2GJ))
#for i in range(N):
#    nu = xsf.sample_blackbody(T, rng.random(5))
#    if nu > bins[-1]:
#        index = len(bin_centers) - 1
#    else:
#        index = np.searchsorted(bins, nu) - 1
#    heights[index] += 1/(bin_widths[index]*N)
#plt.bar(bin_centers*Constants.h, heights, bin_widths[0]*Constants.h, align='center')
#plt.plot(bin_centers*Constants.h, analytic, 'tab:orange', label='Analytic')
#plt.legend()
#plt.show()

# Plot recombination rates over temperature and photoionization cross-sections over energy
M = 1000
T_max = 0.1
T = np.linspace(1e-5, 0.1, M)
mat = xsf.Nitrogen()
n = 2e20
E = np.linspace(1e-3, 0.8, M)
cell_edges = np.linspace(0, 0.7, 100)
recomb_N_per_cell = 10
bremsstrahlung_N_per_level = 2

mesh = Mesh(cell_edges, mat.Z, mat, n, recomb_N_per_cell, bremsstrahlung_N_per_level)
cell = 0
plot_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]

for level in range(mat.Z):
    plt.figure(1)
    rr_rate = np.zeros((M, ))
#    eii_rate = np.zeros((M, ))
#    tbr_rate = np.zeros((M, ))
#    eii_lotz = np.zeros((M, ))
#    tbr_drake = np.zeros((M, ))
    for ind, temp in enumerate(T):
        rr_rate[ind] = mat.rr_n(temp, level + 1) #*level*n**2
#        eii_rate[ind] = mat.sigma_n(temp, level)
#        tbr_rate[ind] = mat.tbr_n(temp, level + 1)
#        U = mat.Eth[level]/temp
#        if level > 4:
#            zeta_Znl = level - 2
#        else:
#            zeta_Znl = mat.Z - level
#        eii_lotz[ind] = 3e-15*zeta_Znl*special.expn(1, U)/U*(temp*1e3)**-1.5
        mesh.ne[cell] = n*(level + 1)
#        tbr_drake[ind] = 4.95e-37*mesh.ne[cell]/(mesh.atom_density[cell]*mat.Eth[level]*1e3*(temp*1e3)**2)*np.exp(U)*special.expn(1, U) # *mesh.atom_density[cell]
    plt.semilogy(T, rr_rate, label="$RR_{"+str(level + 1)+"->"+str(level)+"}$")
    plt.figure(2)
    pi_rate = np.zeros((M, ))
    for ind, energy in enumerate(E):
        pi_rate[ind] = 1e18*mat.pi_n(energy, level) #*n*blackbody(energy/Constants.h, T_max)[0]/energy
    plt.semilogy(E, pi_rate, label="$PI_{"+str(level)+"->"+str(level + 1)+"}$")
    plt.ylim([1e-3, 100])
#    plt.figure(3)
#    plt.semilogy(T, eii_rate, color=plot_colors[level], linestyle='-', label=r"$EII_{" + str(level + 1) + "->" + str(level) + "}$")
#    plt.semilogy(T, eii_lotz, color=plot_colors[level], linestyle='--')
#    plt.figure(4)
#    plt.semilogy(T, tbr_rate, color=plot_colors[level], linestyle='-', label=r"$\alpha_{TBR " + str(level + 1) + "->" + str(level) + "}$")
#    plt.semilogy(T, tbr_drake, color=plot_colors[level], linestyle='--')
#
#    grid_drake = np.zeros((M - 1, M - 1))
#    grid_salz1 = np.zeros((M - 1, M - 1))
#    grid_salz2 = np.zeros((M - 1, M - 1))
#    for i in range(M - 1):
#        for j in range(M - 1):
#            mesh.Te[cell] = T[i] + (T[i + 1] - T[i])*0.5
#            mesh.ni[cell, :] = np.zeros((mat.Z + 1, ))
#            mesh.ni[cell, level + 1] = n
#            mesh.ne[cell] = n*(level + 1)
#            grid_drake[i, j] = mat.ibrem_xs(mesh, cell, E[j]/Constants.h, 0)
#            grid_salz1[i, j] = mat.ibrem_xs(mesh, cell, E[j]/Constants.h, 2)*mesh.ni[cell, level + 1]
#            grid_salz2[i, j] = mat.ibrem_xs(mesh, cell, E[j]/Constants.h, 1)*mesh.ni[cell, level + 1]
#
#    grid_max = max(grid_drake.max(), grid_salz1.max())
#    grid_min = min(grid_drake.min(), grid_salz1.min())
#
#    fig = plt.figure(10 + 3*level)
#    ax = fig.subplots()
#    im = ax.pcolormesh(T, E, grid_drake, norm=colors.LogNorm(vmin=grid_min, vmax=grid_max))
#    fig.colorbar(im, ax=ax)
#    plt.title("Drake Bremsstrahlung for state " + str(level))
#    plt.xlabel("Temperature [keV]")
#    plt.ylabel("Energy [keV]")
#    fig = plt.figure(10 + 3*level + 1)
#    ax = fig.subplots()
#    im = ax.pcolormesh(T, E, grid_salz1, norm=colors.LogNorm(vmin=grid_min, vmax=grid_max))
#    fig.colorbar(im, ax=ax)
#    plt.title("Salzmann 1 Bremsstrahlung for state " + str(level))
#    plt.xlabel("Temperature [keV]")
#    plt.ylabel("Energy [keV]")
#    fig = plt.figure(10 + 3*level + 2)
#    ax = fig.subplots()
#    im = ax.pcolormesh(T, E, grid_salz2, norm=colors.LogNorm(vmin=grid_salz2.min(), vmax=grid_salz2.max()))
#    fig.colorbar(im, ax=ax)
#    plt.title("Salzmann 2 Bremsstrahlung for state " + str(level))
#    plt.xlabel("Temperature [keV]")
#    plt.ylabel("Energy [keV]")

plt.figure(1)
plt.title("Max RR rates over temperature")
plt.xlabel("Temperature [keV]")
plt.ylabel("RR Rate [$cm^{3} ns^{-1}$]")
plt.legend()
plt.figure(2)
plt.title("Photoionization cross sections")
plt.xlabel("Photon Energy [keV]")
plt.ylabel("Photoionization cross section [Mb]")
plt.legend()
#plt.figure(3)
#plt.title("Electron impact ionization rate over temperature - Voronov -- Lotz")
#plt.xlabel("Temperature [keV]")
#plt.ylabel("Electron impact ionization rate [$cm^3 ns^{-1}$]")
#plt.legend()
#plt.figure(4)
#plt.title("Three body recombination rate coefficient over temperature - Voronov -- Lotz")
#plt.xlabel("Temperature [keV]")
#plt.ylabel("Electron impact ionization rate [$cm^6 ns^{-1}$]")
#plt.legend()
#plt.show()

## Test sample from Bremsstrahlung blackbody spectrum at a given temperature
#Te = 0.05
#M = 1000
#mat = xsf.Nitrogen()
#n = 4.3e20
#cell_edges = np.array([0, 0.007])
#recomb_N_per_cell = 10
#N_bremsstrahlung = 1000000
#dt = 1e-3
#
#mesh = Mesh(cell_edges, mat.Z, mat, n, recomb_N_per_cell, N_bremsstrahlung)
#mesh.Te[0] = Te
#mesh.ni[0, :] = n/(mesh.N_levels + 1)*np.ones((mesh.N_levels + 1, ))
#mesh.ne = np.dot(mesh.ni, np.arange(mesh.N_levels + 1))
#
#rng = np.random.default_rng(1)
#
## Create particle census
#type_photon = Particle(0.1, 0.5, 800.0, 1.0, 0, 0.0)
#census = List.empty_list(typeof(type_photon))
#
#energies = np.linspace(1e-4, 1, M + 1)
#energy_centers = energies[:-1] + np.diff(energies)/2
#energy_widths = np.diff(energies)
#analytical = np.zeros((M, ))
#planck = np.zeros((M, ))
#brems_xs = np.zeros((M, ))
#sampled = np.zeros((M, ))
#
#for ind, energy in enumerate(energy_centers):
#    brems_xs[ind] = mat.ibrem_xs(mesh, 0, energy/Constants.h, 2)*np.sum(mesh.ni[0, 1:])
#    analytical[ind] = brems_xs[ind]*xsf.blackbody(energy/Constants.h, Te)[0] #*mesh.cell_widths[0]*dt*4*np.pi/(energy)
#    planck[ind] = xsf.blackbody(energy/Constants.h, Te)[0] #*mesh.cell_widths[0]*dt*4*np.pi/(energy)
#
#census, integrals = mesh.source_particles_bremsstrahlung(rng, census, dt)
#
#print(integrals[0])
#
#for particle in census:
#    if particle.nu*Constants.h > energies[-1]:
#        ind = M - 1
#    elif particle.nu*Constants.h < energies[0]:
#        ind = 0
#    else:
#        ind = np.searchsorted(energies, particle.nu*Constants.h) - 1
#
#    sampled[ind] += particle.w*Constants.h/(energy_widths[0])

# Sample particles from actual distribution
# census = List.empty_list(typeof(type_photon))
# w0 = Constants.sigma_SB*Te**4/M
# denom = integrate.quad(lambda nu : mat.ibrem_xs(mesh, 0, nu, 2)*xsf.blackbody(nu, Te)[0], energies[0]/Constants.h, energies[-1]/Constants.h)[0]
# print(denom)
# for i in range(M):
#     pos_in_cell = mesh.cell_edges[0] + rng.random()*mesh.cell_widths[0]
#     mu = rng.random()*2 - 1
#     start_time = dt*rng.random()
# 
#     xi = rng.random()
#     freq = optimize.root(lambda nu_prime : integrate.quad(lambda nu : mat.ibrem_xs(mesh, 0, nu, 2)*xsf.blackbody(nu, Te)[0], energies[0]/Constants.h, nu_prime)[0] - xi*denom, xi*energies[-1]).x
#     photon = Particle(pos_in_cell, mu, freq, w0, 0, (dt - start_time)/dt*Constants.c*dt)
#     census.append(photon)
# 
# for particle in census:
#     if particle.nu*Constants.h > energies[-1]:
#         ind = M - 1
#     else:
#         ind = np.searchsorted(energies, particle.nu*Constants.h) - 1
#     sampled_integral[ind] += particle.w/(energy_widths[ind]*M)
# 
# print(sampled_integral)

#plt.figure(1)
#plt.plot(energy_centers, brems_xs, label='Inverse brems xs')
#plt.title('Brems xs at T=' + str(Te) + 'keV')
#
#plt.figure(2)
#plt.plot(energy_centers, analytical/integrals[0], color='tab:orange', label='Planck times xs')
#plt.bar(energy_centers, sampled, energies[1] - energies[0], align='center')
#plt.plot(energy_centers, planck*Constants.keV2GJ/(Constants.sigma_SB*Te**4), color='tab:green', label='Planck function')
#plt.legend()
#plt.title('T=' + str(Te) + ' keV')
#
##plt.figure(3)
##plt.plot(energy_centers, analytical/denom, color='tab:orange', label='Planck times xs')
##plt.bar(energy_centers, sampled_integral, energies[1] - energies[0], align='center')
plt.show()