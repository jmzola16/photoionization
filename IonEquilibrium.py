import numpy as np
import scipy
import CrossSectionFunctions as xsf

Te = 0.1                # Electron temperature [keV]
Ts = 0.1                # Source temperature [keV]
a = 0.01372             # Source spectrum [GJ/(cm^3 keV^4)]
c = 30                  # Speed of light [cm/ns]
sigma_SB = a*c/4        # Stefan-Boltzman constant [GJ/(cm^2 ns keV^4)]
GJ2keV = 1/(1.602e-25)  # Conversion factor from GJ to keV

Eph = 2.71*Ts           # Energy per photon [keV]
h = 4.135e-9            # Planck's constant [keV/ns]
flux = 0.5*sigma_SB*Ts**4*GJ2keV/Eph    # Scalar flux of photons [#/(cm^2 ns)]
n = 2e20                # Number density of atoms [#/cm^3]
Eth = np.array([14.53, 29.60, 47.45, 77.47, 113.9])*1e-3   # Threshhold energy [keV]
Emax = np.array([538, 558.1, 584, 614.4, 649.1])*1e-3

gamma = np.array([scipy.integrate.quad(lambda nu : xsf.pi_n1(h*nu)*xsf.blackbody(nu, Ts), Eth[0]/h, Emax[0]/h)[0], 
                  scipy.integrate.quad(lambda nu : xsf.pi_n2(h*nu)*xsf.blackbody(nu, Ts), Eth[1]/h, Emax[1]/h)[0], 
                  scipy.integrate.quad(lambda nu : xsf.pi_n3(h*nu)*xsf.blackbody(nu, Ts), Eth[2]/h, Emax[2]/h)[0], 
                  scipy.integrate.quad(lambda nu : xsf.pi_n4(h*nu)*xsf.blackbody(nu, Ts), Eth[3]/h, Emax[3]/h)[0]])/(a*Ts**4*GJ2keV)

Gamma = np.array([scipy.integrate.quad(lambda nu : xsf.pi_n1(h*nu)*xsf.blackbody(nu, Ts)*(h*nu - Eth[0]), Eth[0]/h, Emax[0]/h)[0], 
                  scipy.integrate.quad(lambda nu : xsf.pi_n2(h*nu)*xsf.blackbody(nu, Ts)*(h*nu - Eth[1]), Eth[1]/h, Emax[1]/h)[0], 
                  scipy.integrate.quad(lambda nu : xsf.pi_n3(h*nu)*xsf.blackbody(nu, Ts)*(h*nu - Eth[2]), Eth[2]/h, Emax[2]/h)[0], 
                  scipy.integrate.quad(lambda nu : xsf.pi_n4(h*nu)*xsf.blackbody(nu, Ts)*(h*nu - Eth[3]), Eth[3]/h, Emax[3]/h)[0]])/(a*Ts**4*GJ2keV)

R = np.array([xsf.rr_n1(Te), xsf.rr_n2(Te), xsf.rr_n3(Te), xsf.rr_n4(Te)])
R_fun = np.array([xsf.rr_n1, xsf.rr_n2, xsf.rr_n3, xsf.rr_n4])

levels = len(Gamma) + 1

ni0 = np.zeros((levels, ))
for i in range(levels - 1):
    ni0[i] = n/(levels)
    ni0[-1] += (i + 1)*ni0[i]

ni1 = np.zeros((levels + 1, ))
ni1[:-1] = ni0
ni1[-1] = 0.5*Ts

def fun(ni):
    F = np.zeros((levels, ))
    F[0] = gamma[0]*flux*(n - sum(ni[0:-1])) - R[0]*ni[0]*ni[-1] - gamma[1]*flux*ni[0] + R[1]*ni[-1]*ni[1]
    F[-1] += ni[0]

    for i in range(1, levels - 2):
        F[i] = gamma[i]*flux*ni[i - 1] - R[i]*ni[i]*ni[-1] - gamma[i + 1]*flux*ni[i] + R[i + 1]*ni[i + 1]*ni[-1]
        F[-1] += (i + 1)*ni[i]

    F[-2] = gamma[-1]*flux*ni[-3] - R[-1]*ni[-2]*ni[-1]
    F[-1] += (levels - 1)*ni[-2]
    F[-1] -= ni[-1]

    return F

def fun_2(ni):
    F = np.zeros((levels + 1, ))
    F[0] = gamma[0]*flux*(n - sum(ni[0:-2])) - R_fun[0](ni[-1])*ni[0]*ni[-2] - gamma[1]*flux*ni[0] + R_fun[1](ni[-1])*ni[-2]*ni[1]
    F[-2] += ni[0]
    F[-1] = Gamma[0]*(n - sum(ni[0:-2]))*flux - R_fun[0](ni[-1])*ni[0]*ni[-2]*(1.5*ni[-1] + Eth[0])

    for i in range(1, levels - 2):
        F[i] = gamma[i]*flux*ni[i - 1] - R_fun[i](ni[-1])*ni[i]*ni[-2] - gamma[i + 1]*flux*ni[i] + R_fun[i + 1](ni[-1])*ni[i + 1]*ni[-2]
        F[-2] += (i + 1)*ni[i]
        F[-1] += Gamma[i]*flux*ni[i - 1] - R_fun[i](ni[-1])*ni[i]*ni[-2]*(1.5*ni[-1] + Eth[i])

    F[-3] = gamma[-1]*flux*ni[-4] - R_fun[-1](ni[-1])*ni[-3]*ni[-2]
    F[-2] += (levels - 1)*ni[-3]
    F[-2] -= ni[-2]
    F[-1] += Gamma[-1]*flux*ni[-4]- R_fun[-1](ni[-1])*ni[-3]*ni[-2]*(1.5*ni[-1] + Eth[i])

    return F

def jac(ni):
    J = np.zeros((levels, levels))

    for i in range(levels - 1):
        J[0, i] -= gamma[0]*flux
        J[-1, i] = i + 1

        if i >= 1:
            J[i, i - 1] += gamma[i]*flux
        if i < levels - 2:
            J[i, i] += -R[i]*ni[-1] - gamma[i + 1]*flux
            J[i, i + 1] += R[i + 1]*ni[-1]
            J[i, -1] += -R[i]*ni[i] + R[i + 1]*ni[i + 1]
        else:
            J[i, i] -= R[i]*ni[-1]
            J[i, -1] -= R[i]*ni[i]

    J[-1, -1] = -1

    K = np.zeros((levels, levels))

    K[0, 0] = -gamma[0]*flux - R[0]*ni[3] - gamma[1]*flux
    K[0, 1] = -gamma[0]*flux + R[1]*ni[3]
    K[0, 2] = -gamma[0]*flux
    K[0, 3] = -ni[0]*R[0] + ni[1]*R[1]
    K[1, 0] = gamma[1]*flux
    K[1, 1] = -gamma[2]*flux - R[1]*ni[3]
    K[1, 2] = R[2]*ni[3]
    K[1, 3] = ni[2]*R[2] - ni[1]*R[1]
    K[2, 0] = 0
    K[2, 1] = gamma[2]*flux
    K[2, 2] = -R[2]*ni[3]
    K[2, 3] = -R[2]*ni[2]
    K[3, 0] = 1
    K[3, 1] = 2
    K[3, 2] = 3
    K[3, 3] = -1

    return J

out = scipy.optimize.root(fun, ni0, jac=jac)

print(out.success)
print(out.x)
print(out.fun)
print(out.message)

sol = scipy.optimize.root(fun_2, ni1)

print(sol.success)
print(sol.x)
print(sol.message)