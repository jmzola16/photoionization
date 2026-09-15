import numpy as np
import Utils.CrossSectionFunctions as xsf
import Utils.Constants as Constants
from MonteCarlo.Mesh import Mesh
from MonteCarlo.Particle import Particle
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.special import roots_laguerre
from scipy.optimize import root
from numba.typed import List
from numba import typeof

mat = xsf.Nitrogen()
N_levels = 7
T_eq = 0.1      # Equilibrium temperature [keV]
n0 = 2e20       # Atom density [cm^-3]
g_q = np.array([4, 1, 2, 1, 2, 1, 2, 1])        # Nitrogen degenerate quantum states

# (1 - np.exp(-Constants.h*nu/T))*

def bb_photoi_rate(T, level, mat=xsf.Nitrogen()):
    return 4*np.pi*quad(lambda E : mat.pi_n(E, level)*xsf.blackbody(E/Constants.h, T)/(E*Constants.h), mat.Eth[level], np.inf)[0]

def bb_photoi_rate_laguerre(T, level, rel_tol=1e-6, max_quad_points=185, mat=xsf.Nitrogen()):
    converged = False

    n_quad_points = 10
    dquad = 10
    derr = 0
    derr_old = 0
    prev_it = 0

    while not converged:
        x_list, wgts = roots_laguerre(n_quad_points)

        photoi_rate = 0
        for ind, x in enumerate(x_list):
            photoi_rate += 4*np.pi*wgts[ind]*mat.pi_n(x, level)/(Constants.h*x)*xsf.blackbody(x/Constants.h, T)*np.exp(x)

        derr = abs(photoi_rate - prev_it)

        if (n_quad_points == max_quad_points) or derr/abs(photoi_rate) < rel_tol:
            converged = True

        if derr_old == 0:
            del_quad_points = dquad
        else:
            del_quad_points = (derr/derr_old)*dquad

        n_quad_points = min(n_quad_points + int(np.ceil(del_quad_points)), max_quad_points)

        derr_old = derr
        prev_it = photoi_rate

    if n_quad_points == max_quad_points:
        print(f"Warning: Maximum quadrature points used to compute integral. Relative error is: {derr/abs(photoi_rate)}")

    return photoi_rate

def integrate_blackbody_laguerre(n_quad_points, T):
    x_list, wgts = roots_laguerre(n_quad_points)
    integral = 0

    nu_list = x_list/Constants.h

    for ind, nu in enumerate(nu_list):
        integral += 4*np.pi*wgts[ind]*xsf.blackbody(nu, T)*np.exp(x_list[ind])/Constants.h

    return integral

def saha_eq(n, T=0.1, n_tot=2e20, mat=xsf.Nitrogen()):
    g_q = np.array([4, 1, 2, 1, 2, 1, 2, 1])
    N_levels = mat.Z
    ne = np.dot(n, np.arange(N_levels + 1))

    out = np.zeros_like(n)

    for level in range(N_levels):
        out[level] = n[level + 1]*ne/n[level] - 2*g_q[level + 1]/g_q[level]*(2*np.pi*Constants.me_keV*T/(Constants.h**2*Constants.c**2))**1.5*np.exp(-mat.Eth[level]/T)

    out[-1] = np.sum(n) - n_tot

    return out

def saha_eq_prime(n, mat=xsf.Nitrogen()):
    N_levels = mat.Z

    n_J = np.zeros((N_levels + 1, N_levels + 1))

    n_J[-1, :] = np.ones((N_levels + 1, ))

    for i in range(N_levels):
        for q in range(N_levels + 1):
            if q == i:
                diff_sum = 0
                for level in range(N_levels):
                    if level != i:
                        diff_sum -= level*n[i + 1]*n[level]/(n[i]**2)

                n_J[i, q] = diff_sum
            elif q == (i + 1):
                diff_sum = 0
                for level in range(N_levels):
                    if level == i:
                        diff_sum += q
                    elif level == i + 1:
                        diff_sum += 2*level*n[i + 1]/n[i]
                    else:
                        diff_sum += level*n[level]/n[i]

                n_J[i, q] = diff_sum
            else:
                n_J[i, q] = q*n[i + 1]/n[i]

    return n_J

## TEST 0: Integrating Planck function
analytic_int = Constants.GJ2keV*Constants.a*Constants.c*T_eq**4
numeric_int = 4*np.pi*quad(lambda nu : xsf.blackbody(nu, T_eq), 0, np.inf)[0]
laguerre_int = integrate_blackbody_laguerre(75, T_eq)

print(f"The analytic integral of the Planck function is {analytic_int} [keV cm^-2 ns^-1]")
print(f"The numerical integral of the Planck function is {numeric_int} [keV cm^-2 ns^-1]")
print(f"The integral of the Planck function using Gauss-Laguerre quadrature is {laguerre_int} [keV cm^-2 ns^-1]")

## TEST 00: Comparing quad to Laguerre quadriture

n_quad_points = 75
x_list, wgts = roots_laguerre(n_quad_points)

for level in range(N_levels):
    skippy_quad = bb_photoi_rate(T_eq, level)
    laguerre_quad = bb_photoi_rate_laguerre(T_eq, level)

    print(f"Level {level} has photoionization rate {skippy_quad:g} with scipy and {laguerre_quad:g} with Laguerre quadrature.")

## TEST 000: Solve Saha equilibrium equation using Newton-Raphson
ns2s = 1e-9
m2cm = 1e2
factor = Constants.me_kg/(Constants.keV2J*ns2s**2*m2cm**2)
print(f"Unit conversions gives a factor of {factor}, whereas me_keV c^-2 is {Constants.me_keV/(Constants.c**2)}")
n_guess = np.arange(1, N_levels + 2)*n0/((N_levels + 1)*(N_levels + 2)/2)
#n_guess = 0.05*n0*np.ones((N_levels + 1, ))/N_levels
#n_guess[0] = 0.95*n0
sol = root(saha_eq, n_guess, jac=saha_eq_prime)

n_nr = sol.x
if sol.success:
    converged = "converged"
else:
    converged = "did not converge"

print(f"NR Saha equation equilibrium level populations {converged} with message '{sol.message}' and are calculated to be: ")
for level in range(N_levels + 1):
    print(f"\tLevel {level}: Pop {n_nr[level]}, Diff {sol.fun[level]} ")

## TEST 0000: Solve Saha equilibrium equation using method in Bowers and Wilson (Ch. 3, pg 104)
const = 2*(2*np.pi*Constants.me_keV/(Constants.h**2*Constants.c**2*1000))**1.5

print(f"My calculated constant is {const} compared to Bowers and Wilson's of 6.088e21")

# Make initial guess for n
ne = 0.5*n0
n_bw_previt = np.zeros((N_levels + 1, ))
n_bw = np.zeros((N_levels + 1, ))
n_bw[0] = 100
for level in range(N_levels):
    n_bw[level + 1] = (n_bw[level]/(n0*(level + 0.5)))*2*(g_q[level + 1]/g_q[level])*(2*np.pi*Constants.me_keV*T_eq/(Constants.h**2*Constants.c**2))**1.5*np.exp(-mat.Eth[level]/T_eq)

# Iterative solve parameters
max_it = 200
rel_tol = 1e-5

# Do iterative solve
it = 0
it_mse = np.max(np.abs(n_bw))
while (it_mse > rel_tol*np.max(np.abs(n_bw))) and it < max_it:
    n_bw_previt = n_bw.copy()
    ne = n0/np.sum(n_bw_previt)*np.dot(n_bw_previt, np.arange(N_levels + 1))

    for level in range(N_levels):
        n_bw[level + 1] = (n_bw_previt[level]/ne)*2*(g_q[level + 1]/g_q[level])*(2*np.pi*Constants.me_keV*T_eq/(Constants.h**2*Constants.c**2))**1.5*np.exp(-mat.Eth[level]/T_eq)

    n_bw[0] = n_bw_previt[0]*np.log10(n0)/np.log10(np.sum(n_bw_previt)) # min(n_bw_previt[0], abs(n0 - np.sum(n_bw[1:]))) #+ n_bw_previt[0]*(n0 - np.sum(n_bw))/(n0 + np.sum(n_bw))

    it_mse = np.sqrt(np.average((n_bw - n_bw_previt)**2))
    it += 1

ne = np.dot(n_bw, np.arange(N_levels + 1))
if it == max_it:
    converged = "did not converge"
else:
    converged = "converged"

print(f"BW Saha equation equilibrium level populations {converged} after {it} iterations with a pseudo-mse of {it_mse} and are calculated to be: ")
for level in range(N_levels + 1):
    print(f"\tLevel {level}: Pop {n_bw[level]}")
print(f"\tTotal: {np.sum(n_bw)}")

## TEST A: Microscopic Spectral Detailed Balance

num_energy_samples = 10000
max_energy = 1
min_energy = np.min(mat.Eth)

rr_rate = np.zeros((N_levels, num_energy_samples))
photoi_rate = np.zeros((N_levels, num_energy_samples))
energy_edges = np.linspace(min_energy, max_energy, num_energy_samples + 1)
energy_widths = np.diff(energy_edges)
energy_samples = energy_edges[:-1] + energy_widths/2

color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

plt.figure()

for level in range(N_levels):
    rr_rate[level, :] = mat.rr_n(T_eq, level + 1)*ne*n_bw[level + 1]*xsf.maxwellian(energy_samples - mat.Eth[level], T_eq)

    for i in range(num_energy_samples):
        photoi_rate[level, i] = 4*np.pi*xsf.integrate_planck_in_number(energy_edges[i], energy_edges[i + 1], T_eq)*mat.pi_n(energy_samples[i], level)*n_bw[level]

    level_diff = photoi_rate[level, 1000] - rr_rate[level, 1000]

    print(f"Level {level} difference between macroscopic recombination rate and photoionization rate at T={energy_samples[1000]} keV in [#/(cm^3 ns)]: {level_diff}")

    plt.semilogy(energy_samples, rr_rate[level, :], c=color_list[level], ls='-', label=f'RR {level + 1}->{level}')
    plt.semilogy(energy_samples, photoi_rate[level, :], c=color_list[level], ls='--', label=f'PI {level}->{level + 1}')

plt.legend()
plt.show()

## TEST B: Saha Equilibrium

def test_saha_eq(end_time, n0, T, print_out=False):
    g_q = np.array([4, 1, 2, 1, 2, 1, 2, 1])
    n = np.zeros((N_levels + 1, ))
    n[:] = np.arange(N_levels + 1, 0, -1)*n0/((N_levels + 1)*(N_levels + 2)/2)
    # n[0] = n0

    bb_photoi_rate_per_level = np.zeros((N_levels, ))

    for level in range(N_levels):
        bb_photoi_rate_per_level[level] = bb_photoi_rate(T, level)

    def dn_dt(t, n):
        rate_matrix = np.zeros((N_levels + 1, N_levels + 1))

        ne = np.dot(n, np.arange(N_levels + 1))
        rate_matrix[0, 0] = -bb_photoi_rate_per_level[0]
        rate_matrix[0, 1] = mat.rr_n(T, 1)*ne
        for level in range(1, N_levels):
            rate_matrix[level, level - 1] = bb_photoi_rate_per_level[level - 1]
            rate_matrix[level, level] = -mat.rr_n(T, level)*ne - bb_photoi_rate_per_level[level]
            rate_matrix[level, level + 1] = mat.rr_n(T, level + 1)*ne

        rate_matrix[N_levels, N_levels - 1] = bb_photoi_rate_per_level[N_levels - 1]
        rate_matrix[N_levels, N_levels] = -mat.rr_n(T, N_levels)*ne

        return np.matmul(rate_matrix, n)

    out = solve_ivp(dn_dt, (0, end_time), n)

    n_out = out.y[:, -1]

    if print_out:
        print(f"Equilibrium Check for T={T} keV: ")
        plt.figure()
    Dq = np.zeros((N_levels))
    ne = np.dot(n_out, np.arange(N_levels + 1))
    for level in range(N_levels):
        saha = 2*g_q[level + 1]/g_q[level]*(2*np.pi*Constants.me_keV*T/(Constants.h**2*Constants.c**2))**1.5*np.exp(-mat.Eth[level]/T)
        Dq[level] = (n_out[level + 1]*ne/n_out[level])/saha
        if print_out:
            print(f"\tLevel {level} Dq: {Dq[level]}")
            print(f"\tSaha Ratio: {saha}")
            print(f"\tIonization level population: {n_out[level]}")
            plt.semilogy(out.t, out.y[level, :], label=str(level))

    if print_out:
        print(f"Total ionization density: {np.sum(n_out)}")
        print(f"Electron density: {ne}")

        plt.legend()
        plt.xlabel("Time [ns]")
        plt.ylabel("Ion Density [$cm^{-3}$]")
        plt.show()

    return Dq

end_time = 1
n0 = 2e20

test_saha_eq(end_time, n0, T_eq, print_out=True)

## TEST C: Temperature Sweep of Saha Test

T_min = 0.02
T_max = 0.1
N_Temps = 10000

Temps = np.linspace(T_min, T_max, N_Temps)

Dq_over_temp = np.zeros((N_levels, N_Temps))

for ind, T in enumerate(Temps):
    Dq_over_temp[:, ind] = test_saha_eq(end_time, n0, T)

for level in range(N_levels):
    plt.plot(Temps, Dq_over_temp[level, :])
plt.xlabel("Temperature [keV]")
plt.ylabel("Dq")
plt.legend([str(level) for level in range(N_levels)])
plt.show()

## TEST D: Relaxation of a Non-Planck Radiation Field

# Define an analogue of advance particles
def advance_particles(census, cutoff, mesh, timestep_dist, *varargs):
    if len(varargs) > 0:
        old_mesh = varargs[0]
    else:
        old_mesh = mesh
    
    mesh.reset_iteration()

    # Move particles through timestep
    index = 0
    leaked_energy = 0.0
    while index < len(census):
        p = census[index]
        # For every particle in the system
        outside_bounds = False
        while p.timestep_dist > 0 and not outside_bounds:
            # Use implicit capture to absorb particle and reduce weight, 
            # accounting for photoionization rate
            outside_bounds = mesh.absorb_particle(p, old_mesh, False, dt)
        
        # Reset the distance each particle has traveled 
        p.timestep_dist = timestep_dist
            
        if p.w < cutoff or outside_bounds:
            # If the particle has a weight below the cutoff or leaves the domain remove it
            leaked_energy += Constants.h*p.nu*p.w
            census.pop(index)
            if p.w < cutoff and not outside_bounds:
                mesh.particle_tally[mesh.get_index(p.cell_i, p.cell_k)] -= 1
        else:
            # Move on to the next particle
            index += 1

    # Calculate recombination
    mesh.calculate_rr_rate(old_mesh, dt)

    return census, mesh, leaked_energy

# Define an analogue of MonteCarlo.MonteCarlo_Modular with constant T
def iterative_solver_const_T(census, cutoff, mesh, timestep_dist, tol, max_it, seed):
    # Define error metrics and max iterations
    it = 0
    err = np.ones((3, ))*np.max(n)
    err_per_it = np.zeros((3, max_it))

    # Copy the initial population which can be reused in each iteration
    population = List()
    for p in census:
        population.append(p.copy())

    # Hold data from previous iterations as convergence criteria
    old_mesh = mesh.copy()
    mesh_previt = mesh.copy()

    while (np.any(abs(err) > tol) and it < max_it):
        census = List()
        for p in population:
            census.append(p.copy())

        # Reset particle tally
        mesh.particle_tally[:] = old_mesh.particle_tally[:]

        # Set the random number seed for generating new particles
        rng = np.random.default_rng(seed)

        census, mesh, leaked_energy = advance_particles(census, cutoff, mesh, timestep_dist, old_mesh)

        # Calculate the update to ionization level and associated error
        mesh.update_state(old_mesh, 0.99)

        mesh.Te[mesh.get_index(0, 0)] = T_eq

        # Use previous iteration's temperatures to source particles from recombination
        census = mesh.source_particles_recombination(rng, census, cutoff, dt)

        err[0] = np.linalg.norm(mesh.ni - mesh_previt.ni)
        err_per_it[0, it] = err[0]
        
        # Calculate the error from flux
        err[1] = np.linalg.norm(mesh.flux - mesh_previt.flux)
        err_per_it[1, it] = err[1]
        
        # Calculate error from material temperature and update
        err[2] = np.linalg.norm(mesh.Te - mesh_previt.Te)
        err_per_it[2, it] = err[2]

        mesh_previt = mesh.copy()

        it += 1

    return census, mesh

dim = 2
N_timesteps = 1000
edges = np.array([0.0, 1.0, 0.0, 1.0])
times = np.linspace(0, end_time, N_timesteps)
dt = end_time/N_timesteps

M = np.array([1, 1], dtype=np.int32)
n = 2e20
N_recomb_tot = 1000
N_bremsstrahlung = 0

cutoff = 1e-12
tol = 1e-4
max_it = 100
bcs = np.ones((2*dim, ), dtype=np.int32)

mesh = Mesh(dim, edges, mat.Z, M, mat, n, N_recomb_tot, N_bremsstrahlung, bcs)

type_photon = Particle(0.1, 0.1, 0.5, np.pi, 800.0, 1.0, 0, 0, 0.0)
census = List.empty_list(typeof(type_photon))

N_source = 10_000
rng = np.random.default_rng()

# Source an initial number of particles
cell_i = 0
cell_k = 0
nu = Constants.GJ2keV*Constants.a*T_eq**4/(N_source*Constants.h)
w = 1.0

for i in range(N_source):
    z_edges = mesh.get_edges_dim(0)
    x_edges = mesh.get_edges_dim(1)

    z_loc = (z_edges[1] - z_edges[0])*rng.random()
    x_loc = (x_edges[1] - x_edges[0])*rng.random()

    mu = 2*rng.random() - 1
    phi = 2*np.pi*rng.random()

    photon = Particle(x_loc, z_loc, mu, phi, nu, w, cell_i, cell_k, Constants.c*dt)

    census.append(photon)

T_rad = np.zeros((N_timesteps, ))

print_time = 0.2
print_int = 0.2

for ind, t in enumerate(times):
    census, mesh = iterative_solver_const_T(census, cutoff, mesh, Constants.c*dt, tol, max_it, ind)

    T_rad[ind] = ((mesh.energy_density[mesh.get_index(cell_i, cell_k)]*Constants.keV2GJ)/Constants.a)**0.25

    if t > print_time:
        print(f"Physical time {t} ns")
        print_time += print_int

plt.figure()
plt.plot(times, T_rad, c='tab:blue', ls='-', label='Rad. Temp.')
plt.plot(times, np.ones_like(times)*T_eq, c='tab:orange', ls='--', label='Mat. Temp.')
plt.legend()

energy_min = 0.001
energy_max = 1
num_energy_bins = 20
energy_bin_edges = np.zeros((num_energy_bins + 1, ))
energy_bin_edges[1:] = np.logspace(energy_min, energy_max, num_energy_bins)
energy_bin_widths = np.diff(energy_bin_edges)
energy_bin_centers = energy_bin_edges[:-1] + energy_bin_widths/2
values = np.zeros_like(energy_bin_centers)

for p in census:
    energy = Constants.h*p.nu
    energy_idx = np.searchsorted(energy_bin_edges, energy)
    if energy > energy_max:
        energy_idx = num_energy_bins - 1

    values[energy_idx] += p.w

blackbody_flux = Constants.GJ2keV*Constants.sigma_SB*T**4

plt.figure()
plt.bar(energy_bin_centers, values/(np.sum(values)*energy_bin_widths), width=energy_bin_widths, align='center')
plt.xlabel("Energy [keV]")
plt.ylabel("Flux")
plt.plot(energy_bin_centers, xsf.blackbody(energy_bin_centers/Constants.h, T_eq)/(Constants.h*blackbody_flux), label='Mat. BB')
plt.show()