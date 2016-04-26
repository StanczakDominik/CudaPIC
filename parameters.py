import numpy as np
from numpy import pi

me = ELECTRON_MASS = 9.10938356e-31
mp = PROTON_MASS = 1.6726219e-27
e = ELECTRON_CHARGE = 1
eps = EPSILON_ZERO = 8.854e-12
k_b = BOLTZMANN_CONSTANT = 1
T = TEMPERATURE = 1*e
N = N_PARTICLES = 128**3
L = DOMAIN_LENGTH = 1e-4
ne = ELECTRON_CONCENTRATION = N/L**3
np = PROTON_CONCENTRATION = ne

lambda_d = DEBYE_LENGTH = (eps*k_b*T/(ne*(-e)**2+np*(e)**2))**0.5
Lambda = PLASMA_PARAMETER = 4 * pi * ne * lambda_d**3
omega_pe = PLASMA_FREQUENCY = (ne*e**2/(me*eps))**0.5  #for COLD electrons
# omega_ce = ELECTRON_CYCLOTRON_FREQUENCY = (e*B/m)
TIME_STEP = 1/(omega_pe/(2*pi))

print("Plasma frequency {:e} rad/s, timestep {:e} s".format(PLASMA_FREQUENCY, TIME_STEP))
print("Debye length {:e} m".format(DEBYE_LENGTH))
print("Plasma parameter {:e}, should be >> 1".format(PLASMA_PARAMETER))
