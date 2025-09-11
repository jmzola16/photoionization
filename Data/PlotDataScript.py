import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./Photoionization')
sys.path.append('..')
sys.path.append('Data')
import CrossSectionFunctions as xsf
import Constants
import scipy

num_cells = 100
cell_edges = np.linspace(0, 0.7, num_cells + 1)
filename = '../Data/MC_Nitrogen_Reemission_t=1.txt'

file = open(filename, 'r')

def read_data(file):
    T = []
    ni = []
    energy_density = []
    t = float(file.readline())
    file.readline()
    energy_density_string = file.readline()

    index = energy_density_string.find(',')
    while index > 0:
        energy_density.append(float(energy_density_string[:index]))
        energy_density_string = energy_density_string[index + 1:]
        index = energy_density_string.find(',')

    file.readline()

    temp_string = file.readline()
    index = temp_string.find(',')
    while index > 0:
        T.append(float(temp_string[:index]))
        temp_string = temp_string[index + 1:]
        index = temp_string.find(',')

    file.readline()

    ni_level_string = file.readline()
    while ni_level_string != '\n':
        level_list = []
        index = ni_level_string.find(',')
        while index > 0:
            level_list.append(float(ni_level_string[:index]))
            ni_level_string = ni_level_string[index + 1:]
            index = ni_level_string.find(',')

        ni.append(level_list)
        ni_level_string = file.readline()

    return t, energy_density, T, ni


t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density)
T = np.array(T)
ni = np.array(ni)

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, label='t='+str(np.round(t, 2)))
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, label='t='+str(t))
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/4.3e20*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, label=str(t))
plt.xlabel('z-location [cm]')
plt.ylabel('\\bar{Z}')
plt.title('Average ionization level over time')

plt.figure(33)
plt.plot(z_loc, ni/4.3e20)
plt.xlabel('z-location [cm]')
plt.ylabel('Ion fraction')
plt.legend(['n'+ str(i) for i in range(len(ni[0, :]))])
plt.title('Ion fraction at t='+ str(np.round(t, 2)))
plt.show()

t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density)
T = np.array(T)
ni = np.array(ni)

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, label='t='+str(np.round(t, 2)))
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, label='t='+str(t))
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/4.3e20*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, label=str(t))
plt.xlabel('z-location [cm]')
plt.ylabel('\\bar{Z}')
plt.title('Average ionization level over time')

t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density)
T = np.array(T)
ni = np.array(ni)

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, label='t='+str(np.round(t, 2)))
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, label='t='+str(t))
plt.legend()
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/4.3e20*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, label=str(t))
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.legend()
plt.title('Average ionization level over time')