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
mat = xsf.Nitrogen()
Eth = np.array([14.53, 29.60, 47.45, 77.47])*1e-3   # Threshhold energy [keV]
Emax = np.array([538, 558.1, 584, 614.4])*1e-3

gamma = np.array([scipy.integrate.quad(lambda nu : mat.pi_n(h*nu, i)*xsf.blackbody(nu, Ts), 
                                       mat.Eth[i]/h, mat.Emax[0]/h)[0] for i in range(4)])/(1/(4*np.pi)*a*c*Ts**4*GJ2keV)

Gamma = np.array([scipy.integrate.quad(lambda nu : mat.pi_n(h*nu, i)*xsf.blackbody(nu, Ts)*(h*nu - Eth[i]), 
                                       mat.Eth[i]/h, mat.Emax[i]/h)[0] for i in range(4)])/(1/(4*np.pi)*a*c*Ts**4*GJ2keV)

R = np.array([mat.rr_n1(Te), mat.rr_n2(Te), mat.rr_n3(Te), mat.rr_n4(Te)])
R_fun = np.array([mat.rr_n1, mat.rr_n2, mat.rr_n3, mat.rr_n4])

levels = len(Gamma) + 1

ni0 = np.zeros((levels, ))
for i in range(levels - 1):
    ni0[i] = n/(levels)
    ni0[-1] += (i + 1)*ni0[i]

ni1 = np.zeros((levels, ))
ni1[-2] = n
ni1[-1] = Ts

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

def fun_2(x):
    F = np.zeros((levels, ))
    ni = np.zeros((levels, ))
    ni[0] = n - sum(x[0:-1])
    ni[1:] = x[0:-1]
    T = x[-1]
    ne = np.sum(ni[1:]*np.arange(1, levels))

    dni = np.zeros((levels - 1, ))
    dT = 0

    dni[0] = gamma[0]*flux*ni[0] - R_fun[0](T)*ni[1]*ne - gamma[1]*flux*ni[1] + R_fun[1](T)*ne*ni[2]
    #F[-2] += ni[0]
    dT = Gamma[0]*ni[0]*flux

    for i in range(1, levels - 2):
        dni[i] = gamma[i]*flux*ni[i] - R_fun[i](T)*ni[i + 1]*ne - gamma[i + 1]*flux*ni[i + 1] + R_fun[i + 1](T)*ni[i + 2]*ne
        #F[-2] += (i + 1)*ni[i]
        dT += Gamma[i]*flux*ni[i] - R_fun[i - 1](T)*ni[i]*ne*(1.5*T + Eth[i])

    dni[-1] = gamma[-1]*flux*ni[-2] - R_fun[-1](T)*ni[-1]*ne
    #F[-2] += (levels - 1)*ni[-3]
    #F[-2] -= ni[-2]
    dT += Gamma[-1]*flux*ni[-2] - R_fun[-2](T)*ni[-2]*ne*(1.5*T + Eth[-2])
    dT -= R_fun[-1](T)*ni[-1]*ne*(1.5*T + Eth[-1])

    dT /= 1.5*(n + ne)

    F[:-1] = dni
    F[-1] = dT

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

print(fun_2(ni1))