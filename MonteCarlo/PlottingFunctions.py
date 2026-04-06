import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
import Constants
import CrossSectionFunctions as xsf

def plot_ionization_level_lineout(mesh, t, loc):
    x_edges = mesh.get_edges_dim(1)
    z_centers = mesh.get_centers_dim(0)
    cell = np.searchsorted(x_edges, loc) - 1
    start = cell*(mesh.cells_per_dim[0])
    end = start + mesh.cells_per_dim[0]

    plt.figure()
    for level in range(mesh.N_levels + 1):
        plt.plot(z_centers, mesh.ni[start:end, level]/mesh.atom_density[start:end])
    plt.xlabel('z-location [cm]')
    plt.ylabel('Ion Fraction [-]')
    plt.legend(['n' + str(i) for i in range(mesh.N_levels + 1)])
    plt.title('Ion fraction at t=' + str(np.round(t, 2)) + ' ns')

def plot_average_ionization_level_lineout(mesh, t, loc):
    if not hasattr(plot_average_ionization_level_lineout, "num_calls") : plot_average_ionization_level_lineout.num_calls = 0
    standard_color_array_matplotlib = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    cell = np.searchsorted(mesh.cell_edges[(mesh.cells_per_dim[0] + 1):], loc) - 1
    start = cell*(mesh.cells_per_dim[0])
    end = start + mesh.cells_per_dim[0]

    plt.figure(11)
    z_bar = np.zeros((mesh.cells_per_dim[0], ))
    for i in range(mesh.N_levels + 1):
        z_bar += i*mesh.ni[start:end, i]/mesh.atom_density[start:end]

    plt.plot(mesh.cell_centers[:mesh.cells_per_dim[0]], z_bar, c=standard_color_array_matplotlib[plot_average_ionization_level_lineout.num_calls], label=str(np.round(t, 2)) + ' ns')
    plt.xlabel('z-location [cm]')
    plt.ylabel(r'$ \overline{Z} $')

    plot_average_ionization_level_lineout.num_calls += 1
    

def plot_average_ionization_level_multicell(mesh, t, dim1, dim2):
    if not hasattr(plot_average_ionization_level_multicell, "num_calls") : plot_average_ionization_level_multicell.num_calls = 0
    standard_color_array_matplotlib = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    start = 0
    end = mesh.cells_per_dim[dim1]
    style = ["-", "--", ":"]
    z_centers = mesh.get_centers_dim(dim1)
    for j in range(mesh.cells_per_dim[dim2]):
        plt.figure(55)
        z_bar = np.zeros((mesh.cells_per_dim[dim1], ))
        for i in range(mesh.N_levels + 1):
            z_bar += i*mesh.ni[start:end, i]/mesh.atom_density[start:end]

        plt.plot(z_centers, z_bar, style[j], c=standard_color_array_matplotlib[plot_average_ionization_level_multicell.num_calls], label=str(np.round(t, 2)) + ' ns')
        plt.xlabel('z-location [cm]')
        plt.ylabel(r'$ \overline{Z} $')

        start += mesh.cells_per_dim[dim1]
        end += mesh.cells_per_dim[dim1]

    plot_average_ionization_level_multicell.num_calls += 1

def plot_radiation_spectrum(mesh, census, x_pos, z_pos, t, plotSpec):
    x_edges = mesh.get_edges_dim(1)
    z_edges = mesh.get_edges_dim(0)
    cell_i = np.searchsorted(x_edges, x_pos) - 1
    cell_k = np.searchsorted(z_edges, z_pos) - 1
    cell_index = mesh.get_index(cell_i, cell_k)
    plt.figure(100 + int(cell_index))

    energy_group_centers = mesh.energy_group_edges[:-1] + np.diff(mesh.energy_group_edges)
    energy_group_widths = np.diff(mesh.energy_group_edges)
    rad_spec = np.zeros((mesh.N_groups, ))

    smooth_N = 100000
    smooth_energy_spectrum = np.linspace(1e-3, 1, smooth_N)
    mat_planck_function = np.zeros((smooth_N, ))
    rad_planck_function = np.zeros((smooth_N, ))
    local_mat_blackbody_tot = Constants.sigma_SB*mesh.Te[cell_index]**4/(Constants.keV2GJ)
    local_rad_blackbody_tot = Constants.c*(mesh.energy_density[cell_index])/(4*np.pi)
    for index, energy in enumerate(smooth_energy_spectrum):
        mat_planck_function[index] = xsf.blackbody(energy/Constants.h, mesh.Te[cell_index])[0]/(Constants.h*local_mat_blackbody_tot)
        if local_rad_blackbody_tot == 0:
            rad_planck_function[index] = 0
        else:
            rad_planck_function[index] = xsf.blackbody(energy/Constants.h, (mesh.energy_density[cell_index]*Constants.keV2GJ/Constants.a)**0.25)[0]/(Constants.h*local_rad_blackbody_tot)


    for particle in census:
        if particle.cell_i == cell_i and particle.cell_k == cell_k:
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

    plt.figure(200 + int(cell_index))
    plt.plot(energy_group_centers, mesh.multigroup_flux[cell_index, :]/(mesh.energy_density[cell_index]*energy_group_widths), label=str(np.round(t, 2))+" ns")
    if plotSpec:
        plt.plot(smooth_energy_spectrum, mat_planck_function, label='Mat. Planck')
        #plt.plot(smooth_energy_spectrum, rad_planck_function, label='Rad. Planck')

    plt.title('Path Length energy spectrum at z=' + str(z_pos))
    plt.xlabel('Energy [keV]')
    plt.ylabel('Normalized Spectrum')
    plt.legend()

def plot_temperatures_lineout(mesh, t, Ts, loc):
    if not hasattr(plot_temperatures_lineout, "num_calls"): plot_temperatures_lineout.num_calls = 0
    standard_color_array_matplotlib = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    x_edges = mesh.get_edges_dim(1)
    z_edges = mesh.get_edges_dim(0)
    z_centers = mesh.get_centers_dim(0)
    cell = np.searchsorted(x_edges, loc) - 1
    start = cell*(mesh.cells_per_dim[0])
    end = start + mesh.cells_per_dim[0]

    plt.figure(22)
    plt.subplot(2, 1, 2)
    plt.plot(z_centers, mesh.Te[start:end], c=standard_color_array_matplotlib[plot_temperatures_lineout.num_calls], label=str(np.round(t, 2))+' ns')
    plt.xlim([z_edges[0], z_edges[-1]])
    plt.ylim([0, Ts])
    plt.xlabel('z-location [cm]')
    plt.ylabel('Material temperature [keV]')
    plt.subplot(2, 1, 1)
    plt.plot(mesh.cell_centers[:mesh.cells_per_dim[0]], (mesh.energy_density[start:end]*Constants.keV2GJ/Constants.a)**0.25, c=standard_color_array_matplotlib[plot_temperatures_lineout.num_calls], label=str(np.round(t, 2))+' ns')
    plt.ylabel('Radiation temperature [keV]')
    plt.xlim([z_edges[0], z_edges[-1]])
    plt.ylim([0, Ts])

    plot_temperatures_lineout.num_calls += 1

def plot_temperatures_multicell(mesh, t, Ts, dim1, dim2):
    if not hasattr(plot_temperatures_multicell, "num_calls") : plot_temperatures_multicell.num_calls = 0
    standard_color_array_matplotlib = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    start = 0
    end = mesh.cells_per_dim[dim1]
    z_centers = mesh.get_centers_dim(dim1)
    z_edges = mesh.get_edges_dim(dim1)
    style = ["-", "--", ":"]
    for i in range(mesh.cells_per_dim[dim2]):
        plt.figure(44)
        plt.subplot(2, 1, 2)
        plt.plot(z_centers, mesh.Te[start:end], style[i], c=standard_color_array_matplotlib[plot_temperatures_multicell.num_calls], label=str(np.round(t, 2))+' ns')
        plt.xlim([z_edges[0], z_edges[-1]])
        plt.ylim([0, Ts])
        plt.xlabel('z-location [cm]')
        plt.ylabel('Material temperature [keV]')
        plt.subplot(2, 1, 1)
        plt.plot(z_centers, (mesh.energy_density[start:end]*Constants.keV2GJ/Constants.a)**0.25, style[i], c=standard_color_array_matplotlib[plot_temperatures_multicell.num_calls], label=str(np.round(t, 2))+' ns')
        plt.ylabel('Radiation temperature [keV]')
        plt.xlim([z_edges[0], z_edges[-1]])
        plt.ylim([0, Ts])

        start += mesh.cells_per_dim[dim1]
        end += mesh.cells_per_dim[dim1]

    plot_temperatures_multicell.num_calls += 1

def plot_alpha_lineout(mesh, t, loc):
    cell = np.searchsorted(mesh.cell_edges[(mesh.cells_per_dim[0] + 1):], loc) - 1
    start = cell*(mesh.cells_per_dim[0])
    end = start + mesh.cells_per_dim[0]

    plt.figure()
    plt.plot(mesh.cell_centers[:mesh.cells_per_dim[0]], mesh.alpha(start, end))
    plt.title(r'$ \alpha $ at t=' + str(round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel(r'$ \alpha $ [-]')
    plt.legend(['n' + str(i) for i in range(mesh.N_levels + 1)])

def write_state_data(mesh, file, t):
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

def write_spectrum_data(mesh, census, file, x_pos, z_pos, t):
    f = open(file, 'a')
    cell_i = np.searchsorted(mesh.cell_edges[(mesh.cells_per_dim[0] + 1):], x_pos) - 1
    cell_k = np.searchsorted(mesh.cell_edges[:(mesh.cells_per_dim[0] + 1)], z_pos) - 1

    f.write('{:d}, {:d}\n'.format(cell_i, cell_k))
    f.write('{:4.2g}'.format(t))
    f.write('\n\n')

    energy_group_widths = np.diff(mesh.energy_group_edges)
    rad_spec = np.zeros((mesh.N_groups, ))

    for particle in census:
        if particle.cell_i == cell_i and particle.cell_k == cell_k:
            if Constants.h*particle.nu > mesh.energy_group_edges[-1]:
                #bin = mesh.N_groups - 1
                continue
            elif Constants.h*particle.nu < mesh.energy_group_edges[0]:
                bin = 0
            else:
                bin = np.searchsorted(mesh.energy_group_edges, Constants.h*particle.nu) - 1
            rad_spec[bin] += particle.w*Constants.h*particle.nu

    rad_spec /= np.sum(rad_spec)*energy_group_widths

    form = ''
    for i in range(mesh.N_groups):
        form += '{:e}, '.format(rad_spec[i])

    f.write(form)
    f.write('\n\n\n')