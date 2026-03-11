import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
import Constants
import CrossSectionFunctions as xsf

def plot_ionization_level(mesh, t):
    plt.figure()
    for level in range(mesh.N_levels + 1):
        plt.plot(mesh.cell_centers, mesh.ni[:, level]/mesh.atom_density)
    plt.xlabel('z-location [cm]')
    plt.ylabel('Ion Fraction [-]')
    plt.legend(['n' + str(i) for i in range(mesh.N_levels + 1)])
    plt.title('Ion fraction at t=' + str(np.round(t, 2)) + ' ns')

def plot_average_ionization_level(mesh, t):
    plt.figure(11)
    z_bar = np.zeros((mesh.N_cells, ))
    for i in range(mesh.N_levels + 1):
        z_bar += i*mesh.ni[:, i]/mesh.atom_density

    plt.plot(mesh.cell_centers, z_bar, label=str(np.round(t, 2)) + ' ns')
    plt.xlabel('z-location [cm]')
    plt.ylabel(r'$ \overline{Z} $')
    
def plot_radiation_spectrum(mesh, census, z_pos, t, plotSpec):
    cell = np.searchsorted(mesh.cell_edges, z_pos) - 1
    plt.figure(100 + int(cell))

    energy_group_centers = mesh.energy_group_edges[:-1] + np.diff(mesh.energy_group_edges)
    energy_group_widths = np.diff(mesh.energy_group_edges)
    rad_spec = np.zeros((mesh.N_groups, ))

    smooth_N = 100000
    smooth_energy_spectrum = np.linspace(1e-3, 1, smooth_N)
    mat_planck_function = np.zeros((smooth_N, ))
    rad_planck_function = np.zeros((smooth_N, ))
    local_mat_blackbody_tot = Constants.sigma_SB*mesh.Te[cell]**4/(Constants.keV2GJ)
    local_rad_blackbody_tot = Constants.c*(mesh.energy_density[cell])/(4*np.pi)
    for index, energy in enumerate(smooth_energy_spectrum):
        mat_planck_function[index] = xsf.blackbody(energy/Constants.h, mesh.Te[cell])[0]/(Constants.h*local_mat_blackbody_tot)
        if local_rad_blackbody_tot == 0:
            rad_planck_function[index] = 0
        else:
            rad_planck_function[index] = xsf.blackbody(energy/Constants.h, (mesh.energy_density[cell]*Constants.keV2GJ/Constants.a)**0.25)[0]/(Constants.h*local_rad_blackbody_tot)


    for particle in census:
        if particle.cell == cell:
            if Constants.h*particle.nu > mesh.energy_group_edges[-1]:
                #bin = mesh.N_groups - 1
                continue
            elif Constants.h*particle.nu < mesh.energy_group_edges[0]:
                bin = 0
            else:
                bin = np.searchsorted(mesh.energy_group_edges, Constants.h*particle.nu) - 1
            rad_spec[bin] += particle.w*Constants.h*particle.nu

    rad_spec /= np.sum(rad_spec)*energy_group_widths

    plt.plot(energy_group_centers, rad_spec, label='t='+str(np.round(t, 2)))
    if plotSpec:
        plt.plot(smooth_energy_spectrum, mat_planck_function, label='Mat. Planck')
        #plt.plot(smooth_energy_spectrum, rad_planck_function, label='Rad. Planck')
    #    print("Te: ", end='')
    #    print(mesh.Te[cell])
    #    print("Tot. blackbody rad. ", end='')
    #    print(local_blackbody_tot)
    #    print("Max Planck (haha): ", end='')
    #    print(xsf.blackbody(2.71*mesh.Te[cell]/Constants.h, mesh.Te[cell])[0])

    plt.title('Energy spectrum at z=' + str(z_pos))
    plt.xlabel('Energy [keV]')
    plt.ylabel('Normalized Spectrum')
    plt.legend()

    plt.figure(200 + int(cell))
    plt.plot(energy_group_centers, mesh.multigroup_flux[cell, :]/(mesh.energy_density[cell]*energy_group_widths), label=str(np.round(t, 2))+" ns")
    if plotSpec:
        plt.plot(smooth_energy_spectrum, mat_planck_function, label='Mat. Planck')
        #plt.plot(smooth_energy_spectrum, rad_planck_function, label='Rad. Planck')

    plt.title('Path Length energy spectrum at z=' + str(z_pos))
    plt.xlabel('Energy [keV]')
    plt.ylabel('Normalized Spectrum')
    plt.legend()

def plot_temperatures(mesh, t, Ts):
    plt.figure(22)
    plt.subplot(2, 1, 2)
    plt.plot(mesh.cell_centers, mesh.Te, label=str(np.round(t, 2))+' ns')
    plt.xlim([mesh.cell_edges[0], mesh.cell_edges[-1]])
    plt.ylim([0, Ts])
    plt.xlabel('z-location [cm]')
    plt.ylabel('Material temperature [keV]')
    plt.subplot(2, 1, 1)
    plt.plot(mesh.cell_centers, (mesh.energy_density*Constants.keV2GJ/Constants.a)**0.25, label=str(np.round(t, 2))+' ns')
    plt.ylabel('Radiation temperature [keV]')
    plt.xlim([mesh.cell_edges[0], mesh.cell_edges[-1]])
    plt.ylim([0, Ts])

def plot_alpha(mesh, t):
    plt.figure()
    plt.plot(mesh.cell_centers, mesh.alpha())
    plt.title(r'$ \alpha $ at t=' + str(round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel(r'$ \alpha $ [-]')
    plt.legend(['n' + str(i) for i in range(mesh.N_levels + 1)])

def write_data(mesh, file, t):
    f = open(file, 'a')
    f.write('{:4.2g}'.format(t))
    f.write('\n\n')
    form = ''
    for i in range(mesh.N_cells):
        form += '{:e}, '.format(mesh.energy_density[i])
    f.write(form)
    f.write('\n\n')
    form = ''
    for i in range(mesh.N_cells):
        form += '{:e}, '.format(mesh.Te[i])
    f.write(form)
    f.write('\n\n')
    for i in range(mesh.N_cells):
        form = ''
        for j in range(mesh.N_levels + 1):
            form += '{:e}, '.format(mesh.ni[i, j])
        f.write(form + '\n')

    f.write('\n')
    f.close()