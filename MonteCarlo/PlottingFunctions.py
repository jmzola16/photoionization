import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
import Constants
import CrossSectionFunctions as xsf

def plot_ionization_level_lineout(mesh, t, dim, loc):
    coords = ['z', 'x']
    plot_centers = mesh.get_centers_dim(dim)
    plot_ni = np.zeros((len(plot_centers), mesh.N_levels + 1))
    plot_atom_density = np.zeros((len(plot_centers), ))

    if dim == 0:
        plot_edges = mesh.get_edges_dim(1)
        cell_i = np.searchsorted(plot_edges, loc) - 1
        start = cell_i*(mesh.cells_per_dim[0])
        end = start + mesh.cells_per_dim[0]
        plot_ni[:, :] = mesh.ni[start:end, :]
        plot_atom_density[:] = mesh.atom_density[start:end]
    elif dim == 1:
        plot_edges = mesh.get_edges_dim(0)
        cell_k = np.searchsorted(plot_edges, loc) - 1
        for cell_i in range(len(plot_centers)):
            plot_ni[cell_i, :] = mesh.ni[mesh.get_index(cell_i, cell_k), :]
            plot_atom_density[cell_i] = mesh.atom_density[cell_i]
    else:
        print("Higher dimensions not yet supported. Aborting...")
        assert 0

    plt.figure()
    for level in range(mesh.N_levels + 1):
        plt.plot(plot_centers, plot_ni[:, level]/plot_atom_density)
    plt.xlabel(coords[dim] + '-location [cm]')
    plt.ylabel('Ion Fraction [-]')
    plt.legend(['n' + str(i) for i in range(mesh.N_levels + 1)])
    plt.title('Ion fraction at t=' + str(np.round(t, 2)) + ' ns')

def plot_average_ionization_level_lineout(mesh, t, dim, loc):
    coords = ['z', 'x']
    if dim == 0 and not hasattr(plot_average_ionization_level_lineout, "num_calls_z") : plot_average_ionization_level_lineout.num_calls_z = 0
    if dim == 1 and not hasattr(plot_average_ionization_level_lineout, "num_calls_x") : plot_average_ionization_level_lineout.num_calls_x = 0
    standard_color_array_matplotlib = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    plot_centers = mesh.get_centers_dim(dim)
    plot_ni = np.zeros((mesh.cells_per_dim[dim], mesh.N_levels + 1))
    plot_atom_density = np.zeros((mesh.cells_per_dim[dim], ))
    if dim == 0: 
        num_calls = plot_average_ionization_level_lineout.num_calls_z
        cell_i = np.searchsorted(mesh.get_edges_dim(1), loc) - 1
        start = cell_i*(mesh.cells_per_dim[0])
        end = start + mesh.cells_per_dim[0]
        plot_ni[:, :] = mesh.ni[start:end, :]
        plot_atom_density[:] = mesh.atom_density[start:end]
    elif dim == 1: 
        num_calls = plot_average_ionization_level_lineout.num_calls_x
        cell_k = np.searchsorted(mesh.get_edges_dim(0), loc) - 1
        for cell_i in range(mesh.cells_per_dim[dim]):
            plot_ni[cell_i, :] = mesh.ni[mesh.get_index(cell_i, cell_k), :]
            plot_atom_density[cell_i] = mesh.atom_density[mesh.get_index(cell_i, cell_k)]
    else: 
        print("Dimensions above 2 not supported. Aborting...")
        assert 0

    plt.figure(11)
    z_bar = np.zeros((mesh.cells_per_dim[dim], ))
    for i in range(mesh.N_levels + 1):
        z_bar += i*plot_ni[:, i]/plot_atom_density[:]

    plt.plot(plot_centers, z_bar, c=standard_color_array_matplotlib[num_calls], label=str(np.round(t, 2)) + ' ns')
    plt.xlabel(coords[dim] + '-location [cm]')
    plt.ylabel(r'$ \overline{Z} $')

    if dim == 0:
        plot_average_ionization_level_lineout.num_calls_z += 1
    elif dim == 1:
        plot_average_ionization_level_lineout.num_calls_x += 1
    

def plot_average_ionization_level_multicell(mesh, t, dim1, dim2):
    if not hasattr(plot_average_ionization_level_multicell, "num_calls_z") and dim1 == 0 : plot_average_ionization_level_multicell.num_calls_z = 0
    if not hasattr(plot_average_ionization_level_multicell, "num_calls_x") and dim1 == 1 : plot_average_ionization_level_multicell.num_calls_x = 0
    standard_color_array_matplotlib = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    style = ["-", "--", ":"]
    plot_centers = mesh.get_centers_dim(dim1)
    for j in range(mesh.cells_per_dim[dim2]):
        z_bar = np.zeros((mesh.cells_per_dim[dim1], ))
        if dim1 == 0:
            num_calls = plot_average_ionization_level_multicell.num_calls_z
            start = j*mesh.cells_per_dim[dim1]
            end = (j + 1)*mesh.cells_per_dim[dim1]
            for i in range(mesh.N_levels + 1):
                z_bar += i*mesh.ni[start:end, i]/mesh.atom_density[start:end]
        elif dim1 == 1:
            num_calls = plot_average_ionization_level_multicell.num_calls_x
            plot_ni = np.zeros((mesh.cells_per_dim[dim1], mesh.N_levels + 1))
            plot_atom_density = np.zeros((mesh.cells_per_dim[dim1], ))
            for k in range(mesh.cells_per_dim[dim1]):
                index = mesh.get_index(k, j)
                plot_ni[k, :] = mesh.ni[index, :]
                plot_atom_density[k] = mesh.atom_density[index]
            for i in range(mesh.N_levels + 1):
                z_bar += i*plot_ni[:, i]/plot_atom_density
        plt.figure(55)
        plt.plot(plot_centers, z_bar, style[j], c=standard_color_array_matplotlib[num_calls], label=str(np.round(t, 2)) + ' ns')
        plt.xlabel('z-location [cm]')
        plt.ylabel(r'$ \overline{Z} $')

    if dim1 == 0:
        plot_average_ionization_level_multicell.num_calls_z += 1
    elif dim1 == 1:
        plot_average_ionization_level_multicell.num_calls_x += 1

def plot_average_ionization_level_heatmap(mesh, t):
    if not hasattr(plot_temperatures_heatmap, "num_calls") : plot_temperatures_heatmap.num_calls = 0
    z_bar = np.zeros((mesh.N_cells, ))
    for i in range(mesh.N_levels + 1):
        z_bar += i*mesh.ni[:, i]/mesh.atom_density
    z_edges = mesh.get_edges_dim(0)
    x_edges = mesh.get_edges_dim(1)
    fig = plt.figure(70 + plot_temperatures_heatmap.num_calls)
    ax = plt.gca()
    im = ax.pcolormesh(z_edges, x_edges, np.reshape(z_bar, (mesh.cells_per_dim[0], mesh.cells_per_dim[1])))
    fig.colorbar(im, ax=ax)
    plt.title("Average Ionization level at " + str(t) + " ns")
    plt.xlabel("z-location")
    plt.ylabel("x-location")

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

def plot_temperatures_lineout(mesh, t, Ts, dim, loc):
    if dim == 0 and not hasattr(plot_temperatures_lineout, "num_calls_z") : plot_temperatures_lineout.num_calls_z = 0
    if dim == 1 and not hasattr(plot_temperatures_lineout, "num_calls_x") : plot_temperatures_lineout.num_calls_x = 0
    standard_color_array_matplotlib = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    plot_centers = mesh.get_centers_dim(dim)
    plot_Te = np.zeros((mesh.cells_per_dim[dim], ))
    plot_energy_density = np.zeros((mesh.cells_per_dim[dim], ))

    if dim == 0:
        plot_x_edges = mesh.get_edges_dim(dim)
        plot_y_edges = mesh.get_edges_dim(1)
        cell_i = np.searchsorted(plot_y_edges, loc) - 1
        start = cell_i*(mesh.cells_per_dim[0])
        end = start + mesh.cells_per_dim[0]
        plot_Te[:] = mesh.Te[start:end]
        plot_energy_density = mesh.energy_density[start:end]
        num_calls = plot_temperatures_lineout.num_calls_z
    elif dim == 1:
        plot_x_edges = mesh.get_edges_dim(dim)
        plot_y_edges = mesh.get_edges_dim(0)
        cell_k = np.searchsorted(plot_y_edges, loc) - 1
        for cell_i in range(mesh.cells_per_dim[dim]):
            plot_Te[cell_i] = mesh.Te[mesh.get_index(cell_i, cell_k)]
            plot_energy_density[cell_i] = mesh.energy_density[mesh.get_index(cell_i, cell_k)]
        num_calls = plot_temperatures_lineout.num_calls_x
    else:
        print("Higher dimensions not yet supported. Aborting...")
        assert 0

    plt.figure(22)
    plt.subplot(2, 1, 2)
    plt.plot(plot_centers, plot_Te, c=standard_color_array_matplotlib[num_calls], label=str(np.round(t, 2))+' ns')
    plt.xlim([plot_x_edges[0], plot_x_edges[-1]])
    plt.ylim([0, 1.5*Ts])
    plt.xlabel('z-location [cm]')
    plt.ylabel('Material temperature [keV]')
    plt.subplot(2, 1, 1)
    plt.plot(plot_centers, (plot_energy_density*Constants.keV2GJ/Constants.a)**0.25, c=standard_color_array_matplotlib[num_calls], label=str(np.round(t, 2))+' ns')
    plt.ylabel('Radiation temperature [keV]')
    plt.xlim([plot_x_edges[0], plot_x_edges[-1]])
    plt.ylim([0, 1.5*Ts])

    if dim == 0:
        plot_temperatures_lineout.num_calls_z += 1
    elif dim == 1:
        plot_temperatures_lineout.num_calls_x += 1

def plot_temperatures_multicell(mesh, t, Ts, dim1, dim2):
    if not hasattr(plot_temperatures_multicell, "num_calls_z") and dim1 == 0 : plot_temperatures_multicell.num_calls_z = 0
    if not hasattr(plot_temperatures_multicell, "num_calls_x") and dim1 == 1 : plot_temperatures_multicell.num_calls_x = 0
    standard_color_array_matplotlib = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    plot_centers = mesh.get_centers_dim(dim1)
    plot_edges = mesh.get_edges_dim(dim1)
    style = ["-", "--", ":"]
    for i in range(mesh.cells_per_dim[dim2]):
        plot_Te = np.zeros((mesh.cells_per_dim[dim1], ))
        plot_energy_density = np.zeros((mesh.cells_per_dim[dim1], ))
        if dim1 == 0:
            num_calls = plot_temperatures_multicell.num_calls_z
            start = i*mesh.cells_per_dim[dim1]
            end = (i + 1)*mesh.cells_per_dim[dim1]
            plot_Te[:] = mesh.Te[start:end]
            plot_energy_density = mesh.energy_density[start:end]
        elif dim1 == 1:
            num_calls = plot_temperatures_multicell.num_calls_x
            for j in range(mesh.cells_per_dim[dim1]):
                index = mesh.get_index(j, i)
                plot_Te[j] = mesh.Te[index]
                plot_energy_density[j] = mesh.energy_density[index]

        plt.figure(44)
        plt.subplot(2, 1, 2)
        plt.plot(plot_centers, plot_Te, style[i], c=standard_color_array_matplotlib[num_calls], label=str(np.round(t, 2))+' ns')
        plt.xlim([plot_edges[0], plot_edges[-1]])
        plt.ylim([0, 1.5*Ts])
        plt.xlabel('z-location [cm]')
        plt.ylabel('Material temperature [keV]')
        plt.subplot(2, 1, 1)
        plt.plot(plot_centers, (plot_energy_density*Constants.keV2GJ/Constants.a)**0.25, style[i], c=standard_color_array_matplotlib[num_calls], label=str(np.round(t, 2))+' ns')
        plt.ylabel('Radiation temperature [keV]')
        plt.xlim([plot_edges[0], plot_edges[-1]])
        plt.ylim([0, 1.5*Ts])

    if dim1 == 0:
        plot_temperatures_multicell.num_calls_z += 1
    if dim1 == 1:
        plot_temperatures_multicell.num_calls_x += 1

def plot_temperatures_heatmap(mesh, t):
    if not hasattr(plot_temperatures_heatmap, "num_calls") : plot_temperatures_heatmap.num_calls = 0
    z_edges = mesh.get_edges_dim(0)
    x_edges = mesh.get_edges_dim(1)
    fig = plt.figure(70 + plot_temperatures_heatmap.num_calls)
    ax = plt.gca()
    im = ax.pcolormesh(z_edges, x_edges, np.reshape(mesh.Te, (mesh.cells_per_dim[0], mesh.cells_per_dim[1])))
    fig.colorbar(im, ax=ax)
    plt.title("Material Temperature [keV] at " + str(t) + " ns")
    plt.xlabel("z-location")
    plt.ylabel("x-location")
    fig = plt.figure(80 + plot_temperatures_heatmap.num_calls)
    ax = plt.gca()
    im = ax.pcolormesh(z_edges, x_edges, np.reshape((mesh.energy_density*Constants.keV2GJ/Constants.a)**0.25, (mesh.cells_per_dim[0], mesh.cells_per_dim[1])))
    fig.colorbar(im, ax=ax)
    plt.title("Radiation Temperature [keV] at " + str(t) + " ns")
    plt.xlabel("z-location")
    plt.ylabel("x-location")

def plot_alpha_lineout(mesh, t, loc):
    if not hasattr(plot_alpha_lineout, "times_called") : plot_alpha_lineout.times_called = 0
    cell = np.searchsorted(mesh.cell_edges[(mesh.cells_per_dim[0] + 1):], loc) - 1
    start = cell*(mesh.cells_per_dim[0])
    end = start + mesh.cells_per_dim[0]

    plt.figure(100 + plot_alpha_lineout.times_called)
    plt.plot(mesh.cell_centers[:mesh.cells_per_dim[0]], mesh.alpha(start, end))
    plt.title(r'$ \alpha $ at t=' + str(round(t, 2)))
    plt.xlabel('z-location [cm]')
    plt.ylabel(r'$ \alpha $ [-]')
    plt.legend(['n' + str(i) for i in range(mesh.N_levels + 1)])

    plot_alpha_lineout.times_called += 1

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