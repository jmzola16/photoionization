import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./Photoionization')
sys.path.append('..')
sys.path.append('Photoionization/Data')
import CrossSectionFunctions as xsf
import Constants
import scipy
import csv

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
filename = 'Photoionization/Data/MC_Gray_10000MaxPart_Nitrogen_Reemission_B_EII_TimeStep0.001_StateData76.txt'

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

def read_wpd_csv(filename):
    datalist = []
    datamat = []
    with open(filename) as csv_file:
        csv_object = csv.reader(csv_file)

        for row in csv_object:
            for point in row:
                datalist.append(float(point))
            datamat.append(np.array(datalist))

            datalist = []

    return np.array(datamat)

gray_rad_temp = []
gray_mat_temp = []
gray_Z_bar = []

gray_mat_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Material Temperature t_0.5.csv'))
gray_mat_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Material Temperature t_1.csv'))
gray_mat_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Material Temperature t_1.5.csv'))
gray_mat_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Material Temperature t_2.csv'))
gray_mat_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Material Temperature t_2.5.csv'))
gray_mat_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Material Temperature t_3.csv'))

gray_rad_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Rad Temp t_0.5.csv'))
gray_rad_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Rad Temp t_1.csv'))
gray_rad_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Rad Temp t_1.5.csv'))
gray_rad_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Rad Temp t_2.csv'))
gray_rad_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Rad Temp t_2.5.csv'))
gray_rad_temp.append(read_wpd_csv('C:/Users/jzola2/Downloads/Rad Temp t_3.csv'))

gray_Z_bar.append(read_wpd_csv('C:/Users/jzola2/Downloads/Z Bar t_0.5.csv'))
gray_Z_bar.append(read_wpd_csv('C:/Users/jzola2/Downloads/Z Bar t_1.csv'))
gray_Z_bar.append(read_wpd_csv('C:/Users/jzola2/Downloads/Z Bar t_1.5.csv'))
gray_Z_bar.append(read_wpd_csv('C:/Users/jzola2/Downloads/Z Bar t_2.csv'))
gray_Z_bar.append(read_wpd_csv('C:/Users/jzola2/Downloads/Z Bar t_2.5.csv'))
gray_Z_bar.append(read_wpd_csv('C:/Users/jzola2/Downloads/Z Bar t_3.csv'))

for data in gray_mat_temp:
    data[:, 0] /= 10
    data[:, 1] /= 1000.0

for data in gray_rad_temp:
    data[:, 0] /= 10
    data[:, 1] /= 1000.0

for data in gray_Z_bar:
    data[:, 0] /= 10

t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density[100:200])
T = np.array(T[100:200])
ni = np.array(ni)[100:200, :]

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, c='tab:blue', ls='-', label='t='+str(np.round(t, 2)))
plt.plot(gray_mat_temp[0][:, 0], gray_mat_temp[0][:, 1], c='tab:blue', ls='--')
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, c='tab:blue', ls='-', label='t='+str(t))
plt.plot(gray_rad_temp[0][:, 0], gray_rad_temp[0][:, 1], c='tab:blue', ls='--')
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, c = 'tab:blue', ls='-', label=str(t) +' ns')
plt.plot(gray_Z_bar[0][:, 0], gray_Z_bar[0][:, 1], c='tab:blue', ls='--')
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
#plt.title('Average ionization level over time')

#file1 = open(filename, 'r')

t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density[100:200])
T = np.array(T[100:200])
ni = np.array(ni)[100:200, :]

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, c='tab:orange', ls='-', label='t='+str(np.round(t, 2)))
plt.plot(gray_mat_temp[1][:, 0], gray_mat_temp[1][:, 1], c='tab:orange', ls='--')
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, c='tab:orange', ls='-', label='t='+str(t))
plt.plot(gray_rad_temp[1][:, 0], gray_rad_temp[1][:, 1], c='tab:orange', ls='--')
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, c='tab:orange', ls='-', label=str(t) +' ns')
plt.plot(gray_Z_bar[1][:, 0], gray_Z_bar[1][:, 1], c='tab:orange', ls='--')
plt.xlabel('z-location [cm]')
plt.ylabel('\\bar{Z}')
#plt.title('Average ionization level over time')

#t, energy_density, T, ni = read_data(file1)
t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density[100:200])
T = np.array(T[100:200])
ni = np.array(ni)[100:200, :]

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, c='tab:green', ls='-', label='t='+str(np.round(t, 2)))
plt.plot(gray_mat_temp[2][:, 0], gray_mat_temp[2][:, 1], c='tab:green', ls='--')
#plt.plot(z_loc, T, label='t='+str(np.round(t, 2)) + ' large ts')
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, c='tab:green', ls='-', label='t='+str(t))
plt.plot(gray_rad_temp[2][:, 0], gray_rad_temp[2][:, 1], c='tab:green', ls='--')
#plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, label='t='+str(t)+' large ts')
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, c='tab:green', ls='-', label=str(t) +' ns')
plt.plot(gray_Z_bar[2][:, 0], gray_Z_bar[2][:, 1], c='tab:green', ls='--')
#plt.plot(z_loc, Z_bar, label=str(t) + ' large ts')
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.legend()
#plt.title('Average ionization level over time')

#t, energy_density, T, ni = read_data(file1)
t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density[100:200])
T = np.array(T[100:200])
ni = np.array(ni)[100:200, :]

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, c='tab:red', ls='-', label='t='+str(np.round(t, 2)))
plt.plot(gray_mat_temp[3][:, 0], gray_mat_temp[3][:, 1], c='tab:red', ls='--')
#plt.plot(z_loc, T, label='t='+str(np.round(t, 2)) + ' large ts')
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, c='tab:red', ls='-', label='t='+str(t))
plt.plot(gray_rad_temp[3][:, 0], gray_rad_temp[3][:, 1], c='tab:red', ls='--')
#plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, label='t='+str(t)+' large ts')
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, c='tab:red', ls='-', label=str(t) +' ns')
plt.plot(gray_Z_bar[3][:, 0], gray_Z_bar[3][:, 1], c='tab:red', ls='--')
#plt.plot(z_loc, Z_bar, label=str(t) + ' large ts')
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.legend()
#plt.title('Average ionization level over time')

t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density[100:200])
T = np.array(T[100:200])
ni = np.array(ni)[100:200, :]

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, c='tab:purple', ls='-', label='t='+str(np.round(t, 2)))
plt.plot(gray_mat_temp[4][:, 0], gray_mat_temp[4][:, 1], c='tab:purple', ls='--')
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, c='tab:purple', ls='-', label='t='+str(t))
plt.plot(gray_rad_temp[4][:, 0], gray_rad_temp[4][:, 1], c='tab:purple', ls='--')
plt.ylabel('Radiation Temperature [keV]')

plt.figure(22)
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, c='tab:purple', ls='-', label=str(t) +' ns')
plt.plot(gray_Z_bar[4][:, 0], gray_Z_bar[4][:, 1], c='tab:purple', ls='--')
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.legend()
#plt.title('Average ionization level over time')

t, energy_density, T, ni = read_data(file)
energy_density = np.array(energy_density[100:200])
T = np.array(T[100:200])
ni = np.array(ni)[100:200, :]

plt.figure(11)
plt.subplot(2, 1, 2)
z_loc = cell_edges[:-1] + np.diff(cell_edges)/2
plt.plot(z_loc, T, c='tab:brown', ls='-', label='t='+str(np.round(t, 2)))
plt.plot(gray_mat_temp[5][:, 0], gray_mat_temp[5][:, 1], c='tab:brown', ls='--')
plt.plot(0.0, 0.0, c='k', ls='--', label='Gray et al.')
plt.xlabel('z-location [cm]')
plt.ylabel('Material temperature [keV]')
plt.xlim([0, 0.7])
plt.ylim([0, 0.15])
plt.legend()
plt.subplot(2, 1, 1)
plt.plot(z_loc, (energy_density/(Constants.a*Constants.GJ2keV))**0.25, c='tab:brown', ls='-', label='t='+str(t))
plt.plot(gray_rad_temp[5][:, 0], gray_rad_temp[5][:, 1], c='tab:brown', ls='--')
plt.ylabel('Radiation Temperature [keV]')
plt.xlim([0, 0.7])
plt.ylim([0, 0.15])

plt.figure(22)
Z_bar = np.sum(ni/n*np.arange(len(ni[0, :])), axis=1)
plt.plot(z_loc, Z_bar, c='tab:brown', ls='-', label=str(t) +' ns')
plt.plot(gray_Z_bar[5][:, 0], gray_Z_bar[5][:, 1], c='tab:brown', ls='--')
plt.plot(0, 0, c='k', ls='--', label='Gray et al.')
plt.xlabel('z-location [cm]')
plt.ylabel('$\\bar{Z}$')
plt.legend()
#plt.title('Average ionization level over time')

plt.figure(33)
plt.plot(z_loc, ni/n)
plt.xlabel('z-location [cm]')
plt.ylabel('Ion fraction')
plt.legend(['n'+ str(i) for i in range(len(ni[0, :]))])
plt.title('Ion fraction at t='+ str(np.round(t, 2)))
plt.show()