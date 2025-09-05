import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import CrossSectionFunctions as xsf

# Simulation parameters
num_cells = 40
num_groups = 340
levels = 7
dt = 1e-4
t_max = 1
Ts = 0.100
Tm0 = np.ones((num_cells, ))*1e-5
Tr0 = 0.1
a = 0.01372
c = 30
keV2GJ = 1.602e-25
GJ2keV = 1/keV2GJ
plot_interval = 0.2
plot_time = 0.2
mat = xsf.Nitrogen()

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
        Z_bar = ni[1:, ].T*np.arange(1, max_levels + 1)
        x_label = "Cell position [cm]"
        legend_unit = " ns"
    else:
        x_variable = varargs[0]
        cell_index = np.searchsorted(cell_edges, independent_variable)
        Z_bar = ni[1:, cell_index, :].T*np.arange(1, max_levels + 1)
        x_label = "Time [ns]"
        legend_unit = " cm"
    
    plt.plot(x_variable, Z_bar, label=str(np.round(independent_variable, 2)) + legend_unit)
    plt.xlabel(x_label)
    plt.ylabel("$\bar{Z}")

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
cell_edges[1] = 5e-6
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
energy_group_centers = energy_group_bounds[0:-1] + np.diff(energy_group_bounds)/2

for i in range(num_groups):
    J0[i] = xsf.integrate_planck_in_energy(energy_group_bounds[i], energy_group_bounds[i + 1], Tr0)

print(sum(J0))
print(a*c*Tr0**4/(4*np.pi)*GJ2keV)

t = 0
Tm = Tm0
ne_old = np.matmul(ni[1:, :].T, np.arange(1, levels + 1))
ne = ne_old.copy()
while t < t_max:
    if t > plot_time:
        plot_average_ionization_level(ni, cell_edges, 7, t)
        plot_electron_temperature(Tm, cell_edges, t)
        plot_time += plot_interval

    Gamma = np.zeros((num_cells, num_groups, levels))
    for i in range(num_cells):
        tau = 0
        for j in range(num_groups):
            for k in range(levels):
                Gamma[i, j, k] = ni[k, i]*mat.pi_n(energy_group_centers[j], k)
                tau += Gamma[i, j, k]*cell_widths[i]

        if i == 0:
            J[i, :] = J0*np.exp(-tau)
        else:
            J[i, :] = J[i - 1, :]*np.exp(-tau)

    dni = np.zeros((levels + 1, num_cells))
    dni[0, :] = ne*mat.rr_n(Tm, 1)*ni[1, :] - np.sum(Gamma[:, :, 0], axis=1)
    for level in range(1, levels):
        dni[level, :] = np.sum(Gamma[:, :, level - 1], axis=1) - ne*mat.rr_n(Tm, level)*ni[level, :] - np.sum(Gamma[:, :, level], axis=1) + ne*mat.rr_n(Tm, level + 1)*ni[level + 1, :]

    dni[levels, :] = np.sum(Gamma[:, :, levels - 1], axis=1) - ne*mat.rr_n(Tm, levels)*ni[levels, :]

    ni += dni

    # Update temperature
    dEabs = np.zeros((num_cells))
    for group in range(len(energy_group_centers)):
        dEabs += np.sum(np.squeeze(Gamma[:, group, :])*np.arange(levels))*J[:, group]

    dEabs *= dt

    dEint = np.zeros((num_cells, ))
    for level in range(levels):
        dEint += dni[level, :]*mat.Eth[level]

    dEint *= dt

    ne = np.matmul(ni[1:, :].T, np.arange(1, levels + 1))
    dne = ne - ne_old

    ne_old = ne.copy()

    dErr = np.zeros((num_cells, ))
    for level in range(1, levels):
        dErr += ne*ni[level, :]*mat.rr_n(Tm, level)*(1.5*Tm + mat.Eth[level - 1])

    Tm += 2/(3*np.sum(ni, axis=0) + ne)*(dEabs - dEint - dErr - 1.5*Tm*dne)

    t += dt

plot_average_ionization_level(ni, cell_edges, 7, t)
plot_electron_temperature(Tm, cell_edges, t)
plt.figure(10)
plt.legend()
plt.figure(20)
plt.legend()
plt.show()