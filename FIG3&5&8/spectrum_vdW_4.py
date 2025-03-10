import os

import numpy as np

from QMClasses import*
from math import*
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors


N      = 4

Omega0 = 6   * 2*pi
Delta  = 18 * 2*pi
T      = 1
B      = 5*Delta
dev    = 1/3 * T
V_min = 0.001
V_max = 1
Steps = 1000




clrs = [(0.7, 0.7, 0.7), 'red']
cm = LinearSegmentedColormap.from_list("Custom", clrs, N=200)



system1 = Hamiltonians(num_of_atoms=4)
system1.chain()
system1.rydberg_3D_array_vdw(block=B)
system1.chirped_linear(delta    =Delta,
                       amplitude=Omega0,
                       t        =T,
                       dev      =T/2.5,
                       omega_const=True)

simulator = QuantumSimulator()
simulator.times = system1.times
vals1, vecs1 = simulator.spectrum(hamiltonian=system1.hamiltonian)     # Calculate E_k(t), |α_k(t)>
detuning = system1.pulses[1](system1.times)

vals1[detuning > 16.1, 3], vals1[detuning > 16.1, 4] = vals1[detuning > 16.1, 4], vals1[detuning > 16.1, 3]
vecs1[detuning > 16.1, :, 3], vecs1[detuning > 16.1, :, 4] = vecs1[detuning > 16.1, :, 4], vecs1[detuning > 16.1, :, 3]
vecs1     = np.einsum('tim, tm->tim', vecs1, np.sign(vecs1[:, 0, :]))
dt       = system1.times[1] - system1.times[0]
der_vecs1 = np.gradient(vecs1, dt, axis=0)
amp1 = np.abs(np.einsum('ti,tim->tm', vecs1[:, :, 0], np.conj(der_vecs1))) ** 2 / Delta


os.makedirs('Data_vdW/N4', exist_ok=True)
np.save('Data_vdW/N4/vals1.npy', vals1)
np.save('Data_vdW/N4/amp1.npy', amp1)



system2 = Hamiltonians(num_of_atoms=4)
system2.chain()
system2.rydberg_3D_array_vdw(block=-B)
system2.chirped_linear(delta    =Delta,
                       amplitude=Omega0,
                       t        =T,
                       dev      =T/2.5,
                       omega_const=True)

simulator = QuantumSimulator()
simulator.times = system2.times
vals2, vecs2 = simulator.spectrum(hamiltonian=system2.hamiltonian)
detuning = system2.pulses[1](system2.times)


vals2[detuning > -16.1, -4], vals2[detuning > -16.1, -5] = vals2[detuning > -16.1, -5], vals2[detuning > -16.1, -4]
vecs2[detuning > -16.1, :, -4], vecs2[detuning > -16.1, :, -5] = vecs2[detuning > -16.1, :, -5], vecs2[detuning > -16.1, :, -4]
vecs2     = np.einsum('tim, tm->tim', vecs2, np.sign(vecs2[:, 0, :]))
dt       = system2.times[1] - system2.times[0]
der_vecs2 = np.gradient(vecs2, dt, axis=0)
amp2 = np.abs(np.einsum('ti,tim->tm', vecs2[:, :, -1], np.conj(der_vecs2)))**2/Delta


np.save('Data_vdW/N4/vals2.npy', vals2)
np.save('Data_vdW/N4/amp2.npy', amp2)
np.save('Data_vdW/N4/detuning.npy', detuning)








