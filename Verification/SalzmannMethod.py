import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import CrossSectionFunctions as xsf
import Constants

# Simulation parameters
num_cells = 40
num_groups = 340
levels = 7
dt = 1e-4
t_max = 1
Ts = 0.100
Tm0 = np.ones((num_cells, ))*1e-5
Tr0 = 0.1
max_it = 1000
use_max_it = True
tot_recomb = np.zeros((num_cells, 100))
plot_interval = 0.2
plot_time = 0.2
mat = xsf.Nitrogen()
dEpi_vec = []
dErr_vec = []
dEint_vec = []

# Plotting function definitions
# Figure codes: 10 - average ionization levels in medium at separate times
#               11 - average ionization level in medium at separate locations
#               20 - electron temperature in medium at separate times
#               21 - electron temperature over time at separate cells

def plot_average_ionization_level(ni, cell_edges, max_levels, independent_variable, *varargs):
    """
    This function plots the average ionization level, either over time or over the domain,
    depending on the given parameters.

    NI - An array of the ionization levels. In the case of plotting over time, this is a 3D
         array organized by [ionization level, position, time]. In the case of plotting 
         over the domain, the time dimension is neglected, giving a 2D array
    CELL_EDGES - the edges of the cells of the spatial domain
    MAX_LEVELS - the number of ionization levels of the material in question
    INDEPENDENT_VARIABLE - either the time at which this plot is taken, or the position in
                           the domain which is to be plotted over time
    VARARGS - In the case of plotting over time, this variable should contain an array of the 
              times being plotted over
    """
    type = len(varargs)
    plt.figure(10 + type)
    if type == 0:
        x_variable = cell_edges[:-1] + np.diff(cell_edges)/2
        Z_bar = np.sum(ni[1:, ].T*np.arange(1, max_levels + 1)/4.3e20, axis=1)
        x_label = "Cell position [cm]"
        legend_unit = " ns"
    else:
        x_variable = varargs[0]
        cell_index = np.searchsorted(cell_edges, independent_variable)
        Z_bar = np.sum(ni[1:, cell_index, :].T*np.arange(1, max_levels + 1)/4.3e20, axis=1)
        x_label = "Time [ns]"
        legend_unit = " cm"
    
    plt.plot(x_variable, Z_bar, label=str(np.round(independent_variable, 2)) + legend_unit)
    plt.xlabel(x_label)
    plt.ylabel("$\\bar{Z}$")

def plot_electron_temperature(Te, cell_edges, independent_variable, *varargs):
    type = len(varargs)
    plt.figure(20 + type)
    if type == 0:
        x_variable = cell_edges[:-1] + np.diff(cell_edges)
        Te_plot = Te
        x_label = "Cell position [cm]"
        legend_unit = " ns"
    else:
        x_variable = varargs[0]
        cell_index = np.searchsorted(cell_edges, independent_variable)
        Te_plot = Te[cell_index, :]
        x_label = "Time [ns]"
        legend_unit = " cm"

    plt.plot(x_variable, Te_plot, label=str(np.round(independent_variable, 2)) + legend_unit)
    plt.xlabel(x_label)
    plt.ylabel("Material Temperature [keV]")

cell_edges = np.zeros((num_cells + 1, ))
cell_edges[1] = 5e-4
for i in range(num_cells - 1):
    cell_edges[i + 2] = cell_edges[i + 1]*1.07

cell_widths = np.diff(cell_edges)
cell_centers = cell_edges[:-1] + cell_widths/2

ni = np.zeros((levels + 1, num_cells))
n0 = 4.3e20*np.ones((num_cells, ))
ni[0, :] = n0

J0 = np.zeros((num_groups, ))
J = np.zeros((num_cells, num_groups))
energy_group_bounds = np.linspace(0.001, 5.000, num_groups + 1)
energy_group_widths = np.diff(energy_group_bounds)
energy_group_centers = energy_group_bounds[0:-1] + energy_group_widths/2

for i in range(num_groups):
    J0[i] = 4*np.pi*xsf.integrate_planck_in_number(energy_group_bounds[i], energy_group_bounds[i + 1], Tr0)

print(sum(J0))
print(16*np.pi*Tr0**3/(Constants.h**3*Constants.c**2)*Constants.zeta3)

t = 0
it = 0
Tm = Tm0.copy()
ne_old = np.matmul(ni[1:, :].T, np.arange(1, levels + 1))
ne = ne_old.copy()
while t < t_max:
    if t > plot_time:
        plot_average_ionization_level(ni, cell_edges, 7, t)
        plot_electron_temperature(Tm, cell_edges, t)
        plot_time += plot_interval

    if use_max_it and it >= max_it:
        break

    Gamma = np.zeros((num_cells, num_groups, levels))
    for i in range(num_cells):
        for j in range(num_groups):
            tau = 0
            for k in range(levels):
                Gamma[i, j, k] = ni[k, i]*mat.pi_n(energy_group_centers[j], k)
                tau += Gamma[i, j, k]*cell_widths[i]

            if i == 0:
                J[i, j] = J0[j]*np.exp(-tau)
            else:
                J[i, j] = J[i - 1, j]*np.exp(-tau)

    dni = np.zeros((levels + 1, num_cells))
    dni[0, :] = dt*(ne*mat.rr_n(Tm, 1)*ni[1, :] - np.sum(Gamma[:, :, 0]*J, axis=1))
    #tot_recomb[:, int(it/10)] += dt*ne*mat.rr_n(Tm, 1)*ni[1, :]
    for level in range(1, levels):
        dni[level, :] = dt*(np.sum(Gamma[:, :, level - 1]*J, axis=1) - ne*mat.rr_n(Tm, level)*ni[level, :] - np.sum(Gamma[:, :, level]*J, axis=1) + ne*mat.rr_n(Tm, level + 1)*ni[level + 1, :])
    #    tot_recomb[:, int(it/10)] += dt*ne*mat.rr_n(Tm, level + 1)*ni[level + 1, :]

    dni[levels, :] = dt*(np.sum(Gamma[:, :, levels - 1]*J, axis=1) - ne*mat.rr_n(Tm, levels)*ni[levels, :])

    ni += dni

    assert np.all(ni >= 0)

    # Update temperature
    dEabs = np.zeros((num_cells, ))
    for group, energy in enumerate(energy_group_centers):
        for level in range(levels):
            dEabs += ni[level, :]*mat.pi_n(energy, level)*J[:, group]*energy_group_widths[group]

    dEabs *= dt

    dEint = np.zeros((num_cells, ))
    for level in range(levels):
        dEint += dni[level + 1, :]*mat.Eth[level]

    dEint *= dt

    ne = np.matmul(ni.T, np.arange(levels + 1))
    dne = ne - ne_old

    ne_old = ne.copy()

    dErr = np.zeros((num_cells, ))
    for level in range(1, levels):
        dErr += ne*ni[level, :]*mat.rr_n(Tm, level)*(1.5*Tm + mat.Eth[level - 1])

    dErr *= dt

    #dEsp = np.zeros((num_cells, ))
    #for cell in range(num_cells):
    #   dEsp[cell] = mat.E_spectral(Tm[cell], ni[:, cell])

    #dEsp *= dt

    Tm += (dEabs - dEint - dErr - 1.5*Tm*dne)/(xsf.cv(Tm, 4.3e20 + ne))

    dEpi_vec.append(dEabs[0])
    dEint_vec.append(dEint[0])
    dErr_vec.append(dErr[0])

    t += dt
    it += 1 

#file = '../Data/Salzmann_Recombination.txt'
#f = open(file, 'w')

#f.write("Recombination in first 100 time steps: \n")
#for i in range(100):
#    for j in range(num_cells):
#        f.write('{:e}, '.format(tot_recomb[j, i]))
#    f.write('\n')

plot_average_ionization_level(ni, cell_edges, 7, t)
plot_electron_temperature(Tm, cell_edges, t)
plt.figure(10)
plt.legend()
plt.figure(20)
plt.legend()
plt.show()

time = np.linspace(0, t_max, len(dEpi_vec))
plt.plot(time, dEpi_vec, label="$dE_{pi}$")
plt.plot(time, dEint_vec, label="$dE_{int}$")
plt.plot(time, dErr_vec, label="$dE_{rr}$")
plt.legend()
plt.show()