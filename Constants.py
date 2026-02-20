# This file contains constants to be used in the rest of the programs
import numpy as np

a = 0.01372                 # Radiation density constant [GJ/(cm^3*keV^4)]
c = 29.97925                # Speed of light in a vacuum [cm/ns]
h = 4.135e-9                # Planck's constant [keV-ns]
sigma_SB = a*c/(4*np.pi)    # Stefan-Boltzmann constant [GJ/(cm^2*keV^4*ster*ns)]
keV2GJ = 1.602e-25          # Conversion factor from keV to GJ
keV2J = 1.602e-16           # Conversion factor from keV to J
GJ2keV = 1/(keV2GJ)         # Conversion factor from GJ to keV
k_B = 8.617e-8              # Boltzmann constant [kev/K]
Na = 6.022e23               # Avogadro's number [#/mol]
zeta3 = 1.2020569031        # Riemann zeta function of 3
me_kg = 9.10938e-31         # Electron rest mass [kg]
me_keV = 511                # Electron rest mass [keV]
eps0 = 8.85418782           # Permittivity of free space [(ns^2 C^2)/(cm^3 kg)]
e = 1.602e-19               # Charge on an electron [C]
mps2cmpns = 1e-7            # Conversion factor from m/s to cm/ns