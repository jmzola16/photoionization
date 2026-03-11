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
spacing = 'linear'
bounds = [0.0, 0.7]
if spacing.lower() == 'linear':
    cell_edges = np.linspace(0, 0.7, num_cells + 1)
elif spacing.lower() == 'loglinear':
    M1 = int(np.floor(num_cells*0.2))
    M2 = num_cells - M1 + 1
    cell_edges = np.zeros((num_cells + 1, ))
    cell_edges[0] = bounds[0]
    cell_edges[1:(M1 + 1)] = np.logspace(-5, np.log(bounds[1]*0.2)/np.log(10), M1)
    cell_edges[M1:] = np.linspace(bounds[1]*0.2, bounds[1], M2)
elif spacing.lower() == 'salzmann':
    cell_edges = np.zeros((num_cells + 1, ))
    cell_edges[1] = 5e-4
    for i in range(num_cells - 1):
        cell_edges[i + 2] = 1.07*cell_edges[i + 1]
#filename = '../Data/MC_Nitrogen_Reemission_Damped.txt'
#filename = '../Data/MC_Nitrogen_Reemission_lam=0.2_NoRR.txt'
#filename = '../Data/MC_Nitrogen_Reemission_lam=0.8_NoRR.txt'
#filename = '../Data/MC_Nitrogen_Reemission_PrevTimeStepRR.txt'
#filename = '../Data/MC_Nitrogen_Reemission_OneEnergy.txt'
#filename = '../Data/MC_Nitrogen_Reemission_LowerEnergyLossRR.txt'
#filename = '../Data/MC_Nitrogen_Reemission_1e6MaxParticles.txt'
#filename = '../Data/MC_Nitrogen_Reemission_lam=0.5_LowEnergyPhoton_IterateRR.txt'
#filename = '../Data/MC_Nitrogen_Reemission_EII_IB.txt'
#filename = '../Data/MC_Nitrogen_Reemission_EII_Subshell_Gray_10.txt'
#filename1 = '../Data/MC_Nitrogen_Reemission_EII_B_Subshell_Salzmann_HighRes_0.txt'
#filename = '../Data/MC_Nitrogen_Reemission_EII_B_Subshell_Salzmann_HighRes_3.txt'
#filename = '../Data/MC_Nitrogen_Reemission_EII_B_Subshell_Salzmann_HighRes_SmallTS_1.txt'
#filename = '../Data/MC_Gray_100000MaxPart_Nitrogen_Reemission_B_EII_TimeStep0.001_1.txt'
#filename = '../Data/MC_Gray_10000MaxPart_Nitrogen_Reemission_B_EII_TimeStep0.001_10.txt'
filename = '../Data/MC_Salzmann_10000MaxPart_Nitrogen_Reemission_B_EII_TimeStep0.001_2.txt'

file = open(filename, 'r')
n = 2e20

def read_data(file):
    T = []
    ni = []
    energy_density = []
    x = file.readline()
    if x == '':
        return 0.0, energy_density, T, ni
    t = float(x)
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
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, label=str(t))
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.title('Average ionization level over time')

#file1 = open(filename, 'r')

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
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, label=str(t))
plt.xlabel('z-location [cm]')
plt.ylabel('\\bar{Z}')
plt.title('Average ionization level over time')

#t, energy_density, T, ni = read_data(file1)
t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density)
T = np.array(T)
ni = np.array(ni)

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, label='t='+str(np.round(t, 2)))
#plt.plot(z_loc, T, label='t='+str(np.round(t, 2)) + ' large ts')
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, label='t='+str(t))
#plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, label='t='+str(t)+' large ts')
plt.legend()
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, label=str(t))
#plt.plot(z_loc, Z_bar, label=str(t) + ' large ts')
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.legend()
plt.title('Average ionization level over time')

#t, energy_density, T, ni = read_data(file1)
t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density)
T = np.array(T)
ni = np.array(ni)

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, label='t='+str(np.round(t, 2)))
#plt.plot(z_loc, T, label='t='+str(np.round(t, 2)) + ' large ts')
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, label='t='+str(t))
#plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, label='t='+str(t)+' large ts')
plt.legend()
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, label=str(t))
#plt.plot(z_loc, Z_bar, label=str(t) + ' large ts')
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.legend()
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
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, label=str(t))
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.legend()
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
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, label=str(t))
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.legend()
plt.title('Average ionization level over time')

plt.figure(33)
plt.plot(z_loc, ni/n)
plt.xlabel('z-location [cm]')
plt.ylabel('Ion fraction')
plt.legend(['n'+ str(i) for i in range(len(ni[0, :]))])
plt.title('Ion fraction at t='+ str(np.round(t, 2)))
plt.show()